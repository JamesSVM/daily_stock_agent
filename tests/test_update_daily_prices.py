from __future__ import annotations

import sqlite3

import pandas as pd

from scripts.update_daily_prices import _ensure_status_table, _get_active_stock_ids, _upsert_stock_history


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


def test_get_active_stock_ids_skips_recently_quarantined_tickers() -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            """
            CREATE TABLE stock_universe (
                stock_id TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                market TEXT NOT NULL,
                transaction_amount REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_status_table(conn)
        conn.executemany(
            """
            INSERT INTO stock_universe
                (stock_id, stock_name, market, transaction_amount, is_active, updated_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            [
                ("2330", "TSMC", "TWSE", 1000000, "2026-08-27T18:30:00+08:00"),
                ("1815", "Unavailable", "TWSE", 100000, "2026-08-27T18:30:00+08:00"),
            ],
        )
        conn.execute(
            """
            INSERT INTO price_update_status
                (stock_id, consecutive_failures, last_error, quarantined, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("1815", 3, "Yahoo returned no usable price data", 1, "2026-08-27T18:30:00+08:00"),
        )

        active = _get_active_stock_ids(conn)

    assert active == ["2330"]
