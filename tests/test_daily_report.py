from __future__ import annotations

from unittest.mock import patch

from daily_report import run


@patch("daily_report.rank_top_candidates")
def test_pipeline_generates_report_without_sending_email(mock_rank, tmp_path):
    mock_rank.return_value = [
        {
            "stock_id": "2330",
            "rank": 1,
            "reason": "test reason",
            "strengths": ["test strength"],
            "risks": ["test risk"],
            "model": "test",
        }
    ]

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_stock_data = {}

    with patch("daily_report.sqlite3.connect", return_value=FakeConn()), patch(
        "daily_report.load_stock_data", return_value=fake_stock_data
    ), patch(
        "daily_report.build_signal_sheet",
        return_value=(
            __import__("pandas").DataFrame(
                [
                    {
                        "date": "2026-08-21",
                        "stock_id": "2330",
                        "market_regime": "bull",
                        "score": 90,
                        "rs20": 0.2,
                        "drawdown_20d": -0.04,
                        "candidate": True,
                        "selected": True,
                        "action": "BUY_NEXT_OPEN",
                    }
                ]
            ),
            "bull",
        ),
    ):
        output = tmp_path / "daily_signal.csv"
        report = run(db_path="unused.db", output_path=str(output), send_notification=False)

    assert "2330" in report
    assert "BUY_NEXT_OPEN" in report
    assert "test reason" in report
    assert "test strength" in report
    assert (tmp_path / "daily_signal.txt").exists()
    mock_rank.assert_called_once()
