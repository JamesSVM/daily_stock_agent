from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from scripts import run_daily_agent
from scripts.update_daily_prices import _ensure_status_table, _get_active_stock_ids, _resolve_expected_date, TAIPEI_TZ


def test_expected_date_falls_back_to_stock_data():
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE daily_price (stock_id TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)"
        )
        conn.executemany(
            "INSERT INTO daily_price(stock_id, date) VALUES (?, ?)",
            [("2330", "2026-08-28"), ("2317", "2026-08-28")],
        )
        _ensure_status_table(conn)

        expected_date = _resolve_expected_date(conn, ["2330", "2317"])

    assert expected_date is not None
    assert expected_date.isoformat() == "2026-08-28"


def test_failure_alert_can_import_email_notifier_from_repo_root(monkeypatch):
    called = {}

    def fake_send_email(body, subject):
        called["subject"] = subject
        called["body"] = body

    monkeypatch.setattr("email_notifier.send_email", fake_send_email)
    run_daily_agent._send_failure_alert(RuntimeError("test failure"))

    assert called["subject"] == "Daily Stock Agent ALERT - Data Refresh Failed"
    assert "test failure" in called["body"]


def test_recently_quarantined_tickers_are_skipped():
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
        recent = (datetime.now(TAIPEI_TZ) - timedelta(days=1)).isoformat()
        conn.execute(
            """
            INSERT INTO price_update_status
                (stock_id, consecutive_failures, last_error, quarantined, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("1815", 3, "Yahoo returned no usable price data", 1, recent),
        )

        active = _get_active_stock_ids(conn)

    assert active == ["2330"]
