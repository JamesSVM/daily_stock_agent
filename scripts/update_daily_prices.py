from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crawler.price import get_price

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_WAIT_SECONDS = 10
DEFAULT_MIN_COVERAGE = 0.90
DEFAULT_DEGRADED_MIN_COVERAGE = 0.75
QUARANTINE_AFTER_FAILURES = 3
QUARANTINE_RETRY_DAYS = 7


def _normalize_stock_id(stock_id: object) -> str:
    return str(stock_id).strip()


def _latest_date(df: pd.DataFrame | None) -> date | None:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date()


def _ensure_status_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_update_status (
            stock_id TEXT PRIMARY KEY,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_success_date TEXT,
            last_error TEXT,
            quarantined INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )


def _get_active_stock_ids(conn: sqlite3.Connection) -> list[str]:
    """Return active stocks, skipping quarantined tickers until their retry date."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_universe'"
    ).fetchone()

    cutoff = (datetime.now(TAIPEI_TZ) - timedelta(days=QUARANTINE_RETRY_DAYS)).isoformat()
    if table_exists:
        rows = conn.execute(
            """
            SELECT u.stock_id
            FROM stock_universe u
            LEFT JOIN price_update_status s ON s.stock_id = u.stock_id
            WHERE u.is_active = 1
              AND (
                  COALESCE(s.quarantined, 0) = 0
                  OR s.updated_at < ?
              )
            ORDER BY u.stock_id
            """,
            (cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT p.stock_id
            FROM daily_price p
            LEFT JOIN price_update_status s ON s.stock_id = p.stock_id
            WHERE COALESCE(s.quarantined, 0) = 0
               OR s.updated_at < ?
            ORDER BY p.stock_id
            """,
            (cutoff,),
        ).fetchall()

    return [_normalize_stock_id(row[0]) for row in rows if row[0] is not None]


def _record_success(conn: sqlite3.Connection, stock_id: str, latest: date | None) -> None:
    now = datetime.now(TAIPEI_TZ).isoformat()
    conn.execute(
        """
        INSERT INTO price_update_status
            (stock_id, consecutive_failures, last_success_date, last_error, quarantined, updated_at)
        VALUES (?, 0, ?, NULL, 0, ?)
        ON CONFLICT(stock_id) DO UPDATE SET
            consecutive_failures=0,
            last_success_date=excluded.last_success_date,
            last_error=NULL,
            quarantined=0,
            updated_at=excluded.updated_at
        """,
        (stock_id, latest.isoformat() if latest else None, now),
    )


def _record_failure(conn: sqlite3.Connection, stock_id: str, error: str) -> int:
    now = datetime.now(TAIPEI_TZ).isoformat()
    conn.execute(
        """
        INSERT INTO price_update_status
            (stock_id, consecutive_failures, last_success_date, last_error, quarantined, updated_at)
        VALUES (?, 1, NULL, ?, 0, ?)
        ON CONFLICT(stock_id) DO UPDATE SET
            consecutive_failures=price_update_status.consecutive_failures + 1,
            last_error=excluded.last_error,
            updated_at=excluded.updated_at
        """,
        (stock_id, error, now),
    )
    failures = conn.execute(
        "SELECT consecutive_failures FROM price_update_status WHERE stock_id = ?",
        (stock_id,),
    ).fetchone()[0]
    if failures >= QUARANTINE_AFTER_FAILURES:
        conn.execute(
            "UPDATE price_update_status SET quarantined=1 WHERE stock_id = ?",
            (stock_id,),
        )
    return int(failures)


def _safe_record_failure(conn: sqlite3.Connection, stock_id: str, error: Exception | str) -> int:
    """Record a ticker failure without turning a secondary DB issue into the root failure."""
    try:
        return _record_failure(conn, stock_id, repr(error))
    except sqlite3.Error as db_error:
        print(f"Price status warning for {stock_id}: {db_error}")
        return 0


def _upsert_stock_history(conn: sqlite3.Connection, stock_id: str, df: pd.DataFrame | None) -> int:
    if df is None or df.empty or "Date" not in df.columns:
        return 0

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["open", "high", "low", "close", "volume"]
    if any(column not in data.columns for column in required):
        return 0

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required)
    if data.empty:
        return 0

    rows = []
    for _, row in data.iterrows():
        rows.append(
            (
                stock_id,
                pd.Timestamp(row["Date"]).date().isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            )
        )

    conn.executemany(
        """
        INSERT INTO daily_price (stock_id, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
        """,
        rows,
    )
    return len(rows)


def _resolve_expected_date(conn: sqlite3.Connection, stock_ids: list[str]) -> date | None:
    """Use the most common latest date across active stock data as the freshness anchor."""
    if not stock_ids:
        return None
    placeholders = ",".join("?" for _ in stock_ids)
    row = conn.execute(
        f"""
        WITH latest_per_stock AS (
            SELECT stock_id, MAX(date) AS latest_date
            FROM daily_price
            WHERE stock_id IN ({placeholders})
            GROUP BY stock_id
        )
        SELECT latest_date, COUNT(*) AS stock_count
        FROM latest_per_stock
        WHERE latest_date IS NOT NULL
        GROUP BY latest_date
        ORDER BY stock_count DESC, latest_date DESC
        LIMIT 1
        """,
        stock_ids,
    ).fetchone()
    if not row or not row[0]:
        return None
    return date.fromisoformat(str(row[0]))


def _coverage(conn: sqlite3.Connection, stock_ids: list[str], expected_date: date) -> tuple[float, list[str]]:
    if not stock_ids:
        return 0.0, []
    placeholders = ",".join("?" for _ in stock_ids)
    rows = conn.execute(
        f"""
        SELECT stock_id, MAX(date) AS latest_date
        FROM daily_price
        WHERE stock_id IN ({placeholders})
        GROUP BY stock_id
        """,
        stock_ids,
    ).fetchall()
    latest_by_stock = {str(stock_id): latest for stock_id, latest in rows}
    expected = expected_date.isoformat()
    stale = [stock_id for stock_id in stock_ids if latest_by_stock.get(stock_id) != expected]
    coverage = (len(stock_ids) - len(stale)) / len(stock_ids)
    return coverage, stale


def _refresh_tickers(
    conn: sqlite3.Connection,
    stock_ids: list[str],
    period: str,
    failures: list[dict[str, object]],
) -> tuple[int, int]:
    stocks_updated = 0
    rows_upserted = 0
    for stock_id in stock_ids:
        try:
            df = get_price(stock_id, period=period)
            count = _upsert_stock_history(conn, stock_id, df)
            if count:
                stocks_updated += 1
                rows_upserted += count
                _record_success(conn, stock_id, _latest_date(df))
            else:
                consecutive = _safe_record_failure(conn, stock_id, "Yahoo returned no usable price data")
                failures.append(
                    {
                        "stock_id": stock_id,
                        "reason": "no_usable_data",
                        "consecutive_failures": consecutive,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - isolate one bad ticker from the batch
            consecutive = _safe_record_failure(conn, stock_id, exc)
            failures.append(
                {
                    "stock_id": stock_id,
                    "reason": repr(exc),
                    "consecutive_failures": consecutive,
                }
            )
    return stocks_updated, rows_upserted


def _write_failure_report(failures: list[dict[str, object]]) -> None:
    path = REPO_ROOT / "reports" / "price_update_failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stock_id", "reason", "consecutive_failures"])
        writer.writeheader()
        writer.writerows(failures)


def _write_quality_status(
    expected_date: date,
    coverage: float,
    stale_ids: list[str],
    degraded: bool,
) -> None:
    path = REPO_ROOT / "reports" / "market_data_status.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "DEGRADED" if degraded else "OK"
    text = (
        f"Status: {status}\n"
        f"Market data date: {expected_date.isoformat()}\n"
        f"Freshness coverage: {coverage:.1%}\n"
        f"Stale/missing stocks: {len(stale_ids)}\n"
    )
    path.write_text(text, encoding="utf-8")


def update_database(
    db_path: str,
    period: str = "3mo",
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_wait_seconds: int = DEFAULT_RETRY_WAIT_SECONDS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    degraded_min_coverage: float = DEFAULT_DEGRADED_MIN_COVERAGE,
    require_today: bool = True,
) -> tuple[int, int]:
    if not 0.0 < degraded_min_coverage <= min_coverage <= 1.0:
        raise ValueError("Coverage thresholds must satisfy 0 < degraded_min_coverage <= min_coverage <= 1.")

    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path, timeout=60) as conn:
        conn.execute("PRAGMA busy_timeout = 60000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_price (
                stock_id TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume INTEGER, PRIMARY KEY (stock_id, date)
            )
            """
        )
        _ensure_status_table(conn)
        stock_ids = _get_active_stock_ids(conn)
        if not stock_ids:
            raise RuntimeError("No active stock_ids found in the stock universe; cannot refresh prices automatically.")

        failures: list[dict[str, object]] = []
        stocks_updated, rows_upserted = _refresh_tickers(conn, stock_ids, period, failures)
        conn.commit()

        today = datetime.now(TAIPEI_TZ).date()
        weekday_requires_today = require_today and today.weekday() < 5

        expected_date = today if weekday_requires_today else _resolve_expected_date(conn, stock_ids)
        if expected_date is None:
            raise RuntimeError("Unable to establish a market-data freshness date from active stock data.")

        # The old implementation re-read SQLite during freshness retry without
        # fetching the stale tickers again. Retry the actual stale tickers instead.
        if weekday_requires_today:
            coverage, stale_ids = _coverage(conn, stock_ids, today)
            for attempt in range(1, retry_attempts + 1):
                if coverage >= min_coverage:
                    break
                print(
                    f"Stock freshness retry {attempt}/{retry_attempts}: "
                    f"coverage={coverage:.1%}, target={min_coverage:.1%}, stale={len(stale_ids)}"
                )
                if coverage >= degraded_min_coverage and attempt == retry_attempts:
                    break
                if attempt == retry_attempts:
                    break
                time.sleep(retry_wait_seconds)
                retry_stocks, retry_rows = _refresh_tickers(conn, stale_ids, period="5d", failures=failures)
                stocks_updated += retry_stocks
                rows_upserted += retry_rows
                conn.commit()
                coverage, stale_ids = _coverage(conn, stock_ids, today)

            if coverage >= min_coverage:
                degraded = False
            elif coverage >= degraded_min_coverage:
                degraded = True
                print(
                    f"Market data warning: degraded freshness coverage {coverage:.1%}; "
                    f"continuing with fresh subset and skipping {len(stale_ids)} stale stocks in signal generation."
                )
            else:
                _write_quality_status(today, coverage, stale_ids, degraded=False)
                _write_failure_report(failures)
                conn.commit()
                raise RuntimeError(
                    f"Market data coverage is too low: {coverage:.1%} < "
                    f"{degraded_min_coverage:.1%}. Stale/missing stocks={len(stale_ids)}. No signal generated."
                )

            expected_date = today
        else:
            coverage, stale_ids = _coverage(conn, stock_ids, expected_date)
            degraded = coverage < min_coverage
            if coverage < degraded_min_coverage:
                _write_quality_status(expected_date, coverage, stale_ids, degraded=False)
                _write_failure_report(failures)
                conn.commit()
                raise RuntimeError(
                    f"Market data coverage is too low: {coverage:.1%} < "
                    f"{degraded_min_coverage:.1%}. Stale/missing stocks={len(stale_ids)}."
                )

        _write_quality_status(expected_date, coverage, stale_ids, degraded)
        _write_failure_report(failures)
        conn.commit()

    print(f"Market data date: {expected_date} (stock-data freshness anchor)")
    print(f"Freshness coverage: {coverage:.1%}")
    print(f"Updated stocks: {stocks_updated}")
    print(f"Upserted rows: {rows_upserted}")
    if degraded:
        print(
            f"Data quality: DEGRADED ({coverage:.1%}); "
            f"signal generation will use stocks with data on {expected_date.isoformat()}."
        )
    if failures:
        print(f"Price update failures: {len(failures)} (see reports/price_update_failures.csv)")
    return stocks_updated, rows_upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh daily_price from Yahoo Finance")
    parser.add_argument("--db", default="data/database.db")
    parser.add_argument("--period", default="3mo")
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--retry-wait-seconds", type=int, default=DEFAULT_RETRY_WAIT_SECONDS)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument(
        "--degraded-min-coverage",
        type=float,
        default=DEFAULT_DEGRADED_MIN_COVERAGE,
        help="Minimum same-day coverage that permits safe degraded-mode signal generation",
    )
    parser.add_argument(
        "--allow-stale-date",
        action="store_true",
        help="Do not require stock-data date to equal today",
    )
    args = parser.parse_args()
    update_database(
        args.db,
        args.period,
        retry_attempts=args.retry_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
        min_coverage=args.min_coverage,
        degraded_min_coverage=args.degraded_min_coverage,
        require_today=not args.allow_stale_date,
    )


if __name__ == "__main__":
    main()
