from __future__ import annotations

import sqlite3

from scripts import run_daily_agent
from scripts.update_daily_prices import _ensure_status_table, _resolve_expected_date


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
