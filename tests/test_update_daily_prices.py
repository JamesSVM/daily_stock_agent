from __future__ import annotations

import sqlite3

import pandas as pd

from scripts.update_daily_prices import _upsert_stock_history


def test_upsert_uses_date_column_not_dataframe_index() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-17"]),
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Volume": [1000],
        }
    )

    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            """
            CREATE TABLE daily_price (
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
        inserted = _upsert_stock_history(conn, "TEST", df)
        row = conn.execute(
            "SELECT stock_id, date, close FROM daily_price"
        ).fetchone()

    assert inserted == 1
    assert row == ("TEST", "2026-08-17", 103.0)
