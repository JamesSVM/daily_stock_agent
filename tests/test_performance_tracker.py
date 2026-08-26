import pandas as pd

from engine.performance_tracker import (
    build_signal_events,
    forward_returns,
    rank_daily_signals,
    summarize_bucket_performance,
    summarize_persistence,
    top_bucket_membership,
)


def test_rank_is_deterministic_and_candidate_first():
    signals = pd.DataFrame(
        [
            {"date": "2026-08-26", "stock_id": "2301", "score": 99, "rs20": 0.30, "candidate": True},
            {"date": "2026-08-26", "stock_id": "2059", "score": 100, "rs20": 0.40, "candidate": True},
            {"date": "2026-08-26", "stock_id": "2330", "score": 100, "rs20": 0.50, "candidate": False},
        ]
    )
    ranked = rank_daily_signals(signals)
    assert ranked.loc[0, "stock_id"] == "2059"
    assert ranked.loc[0, "rank"] == 1
    assert ranked.loc[2, "rank"] == 3


def test_top_bucket_membership_is_cumulative():
    assert top_bucket_membership(1) == ("top_1", "top_3", "top_5", "top_10")
    assert top_bucket_membership(3) == ("top_3", "top_5", "top_10")
    assert top_bucket_membership(5) == ("top_5", "top_10")
    assert top_bucket_membership(10) == ("top_10",)
    assert top_bucket_membership(11) == ()


def test_forward_returns_use_next_open_and_trading_day_horizons():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31"]),
            "open": [100, 110, 120, 121],
            "high": [101, 111, 121, 122],
            "low": [99, 109, 119, 120],
            "close": [105, 115, 125, 130],
            "volume": [1000, 1000, 1000, 1000],
        }
    )
    result = forward_returns(prices, pd.Timestamp("2026-08-26"))
    assert result["entry_date"] == "2026-08-27"
    assert result["entry_open"] == 110
    assert round(result["return_1d"], 6) == round(115 / 110 - 1, 6)
    assert round(result["return_3d"], 6) == round(130 / 110 - 1, 6)


def test_repeated_stock_is_tracked_as_separate_events():
    day1 = pd.DataFrame(
        [{"date": "2026-08-26", "stock_id": "2059", "score": 100, "rs20": 0.5, "rs60": 0.8, "close": 100, "candidate": True, "market_regime": "neutral"}]
    )
    day1 = rank_daily_signals(day1)
    day2 = pd.DataFrame(
        [{"date": "2026-08-27", "stock_id": "2059", "score": 98, "rs20": 0.4, "rs60": 0.7, "close": 110, "candidate": True, "market_regime": "neutral"}]
    )
    day2 = rank_daily_signals(day2)
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31"]),
            "open": [100, 110, 115, 120],
            "high": [101, 111, 116, 121],
            "low": [99, 109, 114, 119],
            "close": [105, 112, 118, 125],
            "volume": [1000, 1000, 1000, 1000],
        }
    )
    events = build_signal_events([day1, day2], {"2059": prices})
    assert len(events) == 2
    assert events["stock_id"].tolist() == ["2059", "2059"]
    persistence = summarize_persistence(events)
    assert persistence.loc[0, "appearances"] == 2


def test_bucket_summary_counts_top3_as_part_of_top5_and_top10():
    events = pd.DataFrame(
        [
            {"signal_date": "2026-08-26", "stock_id": "2059", "rank": 1, "return_1d": 0.10, "excess_return_1d": 0.08, "return_3d": 0.05, "excess_return_3d": 0.03, "return_5d": 0.06, "excess_return_5d": 0.04, "return_10d": 0.02, "excess_return_10d": 0.01},
            {"signal_date": "2026-08-26", "stock_id": "3653", "rank": 3, "return_1d": -0.02, "excess_return_1d": -0.03, "return_3d": 0.01, "excess_return_3d": 0.00, "return_5d": 0.03, "excess_return_5d": 0.01, "return_10d": 0.04, "excess_return_10d": 0.02},
            {"signal_date": "2026-08-26", "stock_id": "3443", "rank": 5, "return_1d": 0.04, "excess_return_1d": 0.02, "return_3d": 0.02, "excess_return_3d": 0.01, "return_5d": 0.01, "excess_return_5d": 0.00, "return_10d": 0.00, "excess_return_10d": -0.01},
        ]
    )
    summary = summarize_bucket_performance(events)
    top3 = summary[(summary["bucket"] == "top_3") & (summary["horizon"] == "1d")].iloc[0]
    top5 = summary[(summary["bucket"] == "top_5") & (summary["horizon"] == "1d")].iloc[0]
    assert top3["samples"] == 2
    assert top5["samples"] == 3
