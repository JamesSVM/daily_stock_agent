import pandas as pd

from engine.performance_tracker import forward_returns


def test_forward_returns_handles_missing_price_frame():
    result = forward_returns(None, pd.Timestamp("2026-08-27"))
    assert result["entry_date"] is None
    assert result["entry_open"] is None
    assert result["return_1d"] is None
    assert result["return_10d"] is None


def test_forward_returns_handles_frame_without_date_column():
    prices = pd.DataFrame({"open": [100.0], "close": [101.0]})
    result = forward_returns(prices, pd.Timestamp("2026-08-27"))
    assert result["entry_date"] is None
