from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crawler.price import get_price

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_WAIT_SECONDS = 600
DEFAULT_MIN_COVERAGE = 0.90
QUARANTINE_AFTER_FAILURES = 3
BENCHMARK_SYMBOL = "^TWII"


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
    """Return the current universe; keep the old DB fallback for safe upgrades."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='stock_universe'"
    ).fetchone()

    if table_exists:
        rows = conn.execute(
            """
            SELECT stock_id
            FROM stock_universe
            WHERE is_active = 1
            ORDER BY stock_id
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT p.stock_id
            FROM daily_price p
            LEFT JOIN price_update_status s ON s.stock_id = p.stock_id
            WHERE COALESCE(s.quarantined, 0) = 0
            ORDER BY p.stock_id
            """
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


def _upsert_stock_history(conn: sqlite3.Connection, stock_id: str, df: pd.DataFrame) -> int:
    if df is None or df.empty or "Date" not in df.columns:
        return 0

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

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
        trade_date = pd.Timestamp(row["Date"]).date().isoformat()
        rows.append((
            stock_id,
            trade_date,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
        ))

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


def _fetch_benchmark_date() -> date:
    benchmark = get_price(BENCHMARK_SYMBOL, period="5d")
    latest = _latest_date(benchmark)
    if latest is None:
        raise RuntimeError(f"Unable to fetch benchmark {BENCHMARK_SYMBOL} from Yahoo Finance.")
    return latest


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


def _write_failure_report(failures: list[dict[str, object]]) -> None:
    path = REPO_ROOT / "reports" / "price_update_failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stock_id", "reason", "consecutive_failures"])
        writer.writeheader()
        writer.writerows(failures)


def update_database(
    db_path: str,
    period: str = "3mo",
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_wait_seconds: int = DEFAULT_RETRY_WAIT_SECONDS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    require_today: bool = True,
) -> tuple[int, int]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
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
                    consecutive = _record_failure(conn, stock_id, "Yahoo returned no usable price data")
                    failures.append({
                        "stock_id": stock_id,
                        "reason": "no_usable_data",
                        "consecutive_failures": consecutive,
                    })
            except Exception as exc:  # noqa: BLE001 - isolate one bad ticker from the batch
                consecutive = _record_failure(conn, stock_id, repr(exc))
                failures.append({
                    "stock_id": stock_id,
                    "reason": repr(exc),
                    "consecutive_failures": consecutive,
                })

        conn.commit()

        expected_date = _fetch_benchmark_date()
        today = datetime.now(TAIPEI_TZ).date()
        if require_today and today.weekday() < 5 and expected_date != today:
            for attempt in range(1, retry_attempts + 1):
                print(
                    f"Freshness retry {attempt}/{retry_attempts}: "
                    f"Yahoo benchmark date={expected_date}, expected={today}"
                )
                if attempt < retry_attempts:
                    time.sleep(retry_wait_seconds)
                    expected_date = _fetch_benchmark_date()
                    if expected_date == today:
                        break
            if expected_date != today:
                raise RuntimeError(
                    f"Market data is stale: Yahoo {BENCHMARK_SYMBOL} latest date="
                    f"{expected_date}, expected={today}. No signal generated."
                )

        coverage, stale_ids = _coverage(conn, stock_ids, expected_date)
        for attempt in range(1, retry_attempts + 1):
            if coverage >= min_coverage:
                break
            print(
                f"Stock freshness retry {attempt}/{retry_attempts}: "
                f"coverage={coverage:.1%}, target={min_coverage:.1%}, stale={len(stale_ids)}"
            )
            if attempt == retry_attempts:
                break
            time.sleep(retry_wait_seconds)
            for stock_id in stale_ids:
                try:
                    df = get_price(stock_id, period="5d")
                    count = _upsert_stock_history(conn, stock_id, df)
                    if count:
                        _record_success(conn, stock_id, _latest_date(df))
                    else:
                        _record_failure(conn, stock_id, "Retry returned no usable price data")
                except Exception as exc:  # noqa: BLE001
                    _record_failure(conn, stock_id, repr(exc))
            conn.commit()
            coverage, stale_ids = _coverage(conn, stock_ids, expected_date)

        _write_failure_report(failures)
        conn.commit()

        if coverage < min_coverage:
            raise RuntimeError(
                f"Market data coverage is too low: {coverage:.1%} < {min_coverage:.1%}. "
                f"Stale/missing stocks={len(stale_ids)}. No signal generated."
            )

    print(f"Benchmark date: {expected_date}")
    print(f"Freshness coverage: {coverage:.1%}")
    print(f"Updated stocks: {stocks_updated}")
    print(f"Upserted rows: {rows_upserted}")
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
    parser.add_argument("--allow-stale-date", action="store_true", help="Do not require Yahoo benchmark date to equal today")
    args = parser.parse_args()
    update_database(
        args.db,
        args.period,
        retry_attempts=args.retry_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
        min_coverage=args.min_coverage,
        require_today=not args.allow_stale_date,
    )


if __name__ == "__main__":
    main()
