import pandas as pd
from engine.performance_tracker import forward_returns

def test_forward_returns_missing_frame_is_safe():
    result = forward_returns(None, pd.Timestamp("2026-08-27"))
    assert result["entry_date"] is None
    assert result["return_1d"] is None
