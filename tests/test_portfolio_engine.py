import pytest
from datetime import date

from engine.portfolio_engine import (
    PortfolioConfig,
    PortfolioState,
    allocate_candidate,
    apply_portfolio_decisions,
)


def configured_state(**overrides) -> PortfolioState:
    values = {
        "total_capital": 100_000,
        "cash_available": 100_000,
        "max_positions": 5,
        "max_position_pct": 0.20,
        "target_position_pct": 0.15,
        "max_daily_trades": 1,
        "max_monthly_trades": 10,
        "min_score": 85,
        "min_order_amount": 1_000,
    }
    values.update(overrides)
    return PortfolioState(config=PortfolioConfig(**values), configured=True)


def test_missing_portfolio_state_blocks_buy():
    state = PortfolioState(config=PortfolioConfig(), configured=False)
    result = allocate_candidate(
        stock_id="2330",
        score=99,
        close=1000,
        market_regime="BULL",
        signal_date=date(2026, 9, 1),
        state=state,
        trades_today=0,
    )
    assert result["action"] == "NO_TRADE"
    assert result["portfolio_reason"] == "portfolio_state_not_configured"


def test_buy_is_budgeted_and_limited_to_one_trade_per_day():
    state = configured_state()
    result = allocate_candidate(
        stock_id="2330",
        score=95,
        close=1000,
        market_regime="BULL",
        signal_date=date(2026, 9, 1),
        state=state,
        trades_today=0,
    )
    assert result["action"] == "BUY_NEXT_OPEN"
    assert result["selected"] is True
    assert result["allocation_amount"] == pytest.approx(15_000)

    blocked = allocate_candidate(
        stock_id="3017",
        score=96,
        close=500,
        market_regime="BULL",
        signal_date=date(2026, 9, 1),
        state=state,
        trades_today=1,
    )
    assert blocked["selected"] is False
    assert blocked["portfolio_reason"] == "daily_trade_limit_reached"


def test_held_stock_becomes_hold_not_buy():
    state = configured_state()
    state.positions["2330"] = 15_000
    result = allocate_candidate(
        stock_id="2330",
        score=99,
        close=1000,
        market_regime="BULL",
        signal_date=date(2026, 9, 1),
        state=state,
        trades_today=0,
    )
    assert result["action"] == "HOLD"
    assert result["selected"] is False


def test_bear_market_blocks_new_entries_by_default():
    state = configured_state()
    result = allocate_candidate(
        stock_id="2330",
        score=99,
        close=1000,
        market_regime="BEAR",
        signal_date=date(2026, 9, 1),
        state=state,
        trades_today=0,
    )
    assert result["action"] == "NO_TRADE"
    assert result["portfolio_reason"] == "bear_market_entry_block"


def test_apply_decisions_only_selects_highest_candidate():
    import pandas as pd

    state = configured_state()
    signals = pd.DataFrame(
        [
            {"stock_id": "2330", "candidate": True, "score": 95, "rs20": 0.7, "close": 1000, "market_regime": "BULL"},
            {"stock_id": "3017", "candidate": True, "score": 93, "rs20": 0.6, "close": 500, "market_regime": "BULL"},
            {"stock_id": "2059", "candidate": False, "score": 99, "rs20": 0.8, "close": 300, "market_regime": "BULL"},
        ]
    )
    result = apply_portfolio_decisions(signals, state, signal_date=date(2026, 9, 1))
    selected = result[result["selected"]]
    assert selected["stock_id"].tolist() == ["2330"]
    assert result.loc[result["stock_id"] == "3017", "portfolio_reason"].iloc[0] == "daily_trade_limit_reached"
    assert result.loc[result["stock_id"] == "2059", "portfolio_reason"].iloc[0] == "not_in_top_candidate_set"
