from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from crawler.price import get_price


def _normalize_stock_id(stock_id: object) -> str:
    return str(stock_id).strip()


def _upsert_stock_history(
    conn: sqlite3.Connection,
    stock_id: str,
    df: pd.DataFrame,
) -> int:
    if df is None or df.empty:
        return 0

    data = df.copy()
    if "Date" not in data.columns:
        return 0

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename)

    required = ["open", "high", "low", "close", "volume"]
    if any(column not in data.columns for column in required):
        return 0

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required)
    if data.empty:
        return 0

    rows = [
        (
            stock_id,
            timestamp.date().isoformat(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
        )
        for timestamp, row in data.iterrows()
    ]

    conn.executemany(
        """
        INSERT INTO daily_price
            (stock_id, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume
        """,
        rows,
    )
    return len(rows)


def update_database(db_path: str, period: str = "3mo") -> tuple[int, int]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_price (
                stock_id TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (stock_id, date)
            )
            """
        )
        stock_ids = [
            _normalize_stock_id(row[0])
            for row in conn.execute(
                "SELECT DISTINCT stock_id FROM daily_price ORDER BY stock_id"
            ).fetchall()
            if row[0] is not None
        ]

        if not stock_ids:
            raise RuntimeError("No stock_ids found in daily_price; cannot refresh prices automatically.")

        stocks_updated = 0
        rows_upserted = 0
        for stock_id in stock_ids:
            df = get_price(stock_id, period=period)
            count = _upsert_stock_history(conn, stock_id, df)
            if count:
                stocks_updated += 1
                rows_upserted += count

        conn.commit()

    return stocks_updated, rows_upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh daily_price from Yahoo Finance")
    parser.add_argument("--db", default="data/database.db")
    parser.add_argument("--period", default="3mo")
    args = parser.parse_args()

    stocks, rows = update_database(args.db, args.period)
    print(f"Updated stocks: {stocks}")
    print(f"Upserted rows: {rows}")


if __name__ == "__main__":
    main()
