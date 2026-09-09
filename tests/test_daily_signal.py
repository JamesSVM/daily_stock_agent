from __future__ import annotations

import numpy as np
import pandas as pd

from daily_signal import (
    MIN_SIGNAL_SCORE,
    PULLBACK_MAX,
    PULLBACK_MIN,
    RS_THRESHOLD,
    build_signal_sheet,
    is_v15_candidate,
)
from engine.portfolio_engine import PortfolioConfig, PortfolioState, decide_action


def _make_stock(close_values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=len(close_values))
    close = np.asarray(close_values, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000.0),
        },
        index=dates,
    )


def _synthetic_universe() -> dict[str, pd.DataFrame]:
    n = 90
    weak = np.linspace(100.0, 100.5, n).tolist()
    strong = list(np.linspace(70.0, 90.0, 70))
    strong.extend(np.linspace(90.0, 110.0, 10))
    strong.extend(np.linspace(110.0, 105.0, 10))
    return {
        "STRONG": _make_stock(strong),
        "WEAK1": _make_stock(weak),
        "WEAK2": _make_stock(weak),
        "WEAK3": _make_stock(weak),
    }


def test_v15_threshold_constants_are_frozen() -> None:
    assert RS_THRESHOLD == 0.10
    assert PULLBACK_MIN == -0.07
    assert PULLBACK_MAX == -0.02
    assert MIN_SIGNAL_SCORE == 70.0


def test_entry_candidate_rule_is_regime_independent() -> None:
    assert is_v15_candidate(
        rs20=0.15,
        drawdown_20d=-0.05,
        score=80.0,
        trend_pass=True,
        momentum_pass=True,
    )


def _state(tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    state_path.write_text(
        '{"config": {"total_capital": 300000, "min_score": 85, "bear_market_entry": false}, "positions": {}}',
        encoding="utf-8",
    )
    return state_path


def test_daily_signal_returns_latest_date_and_expected_columns(tmp_path) -> None:
    signals, regime = build_signal_sheet(_synthetic_universe(), portfolio_state_path=_state(tmp_path))
    assert not signals.empty
    assert signals["date"].nunique() == 1
    assert regime in {"bull", "neutral", "bear"}
    expected = {
        "date", "stock_id", "market_regime", "close", "rs20", "rs60",
        "drawdown_20d", "score", "trend_pass", "momentum_pass",
        "pullback_pass", "candidate", "selected", "action", "reason",
        "trade_score", "portfolio_reason", "total_capital", "position_count",
    }
    assert expected.issubset(signals.columns)


def test_held_stock_is_hold(tmp_path) -> None:
    state_path = tmp_path / "portfolio_state.json"
    state_path.write_text(
        '{"config": {"total_capital": 300000, "min_score": 85, "bear_market_entry": false}, "positions": {"STRONG": 1}}',
        encoding="utf-8",
    )
    signals, _ = build_signal_sheet(_synthetic_universe(), portfolio_state_path=state_path)
    row = signals.loc[signals["stock_id"] == "STRONG"].iloc[0]
    assert bool(row["candidate"])
    assert row["action"] == "HOLD"
    assert not bool(row["selected"])


def test_below_minimum_score_is_watch() -> None:
    state = PortfolioState(
        config=PortfolioConfig(total_capital=300000, min_score=85),
        configured=True,
    )
    decision = decide_action(stock_id="2330", score=84.9, market_regime="BULL", state=state)
    assert decision["action"] == "WATCH"
    assert decision["portfolio_reason"] == "below_minimum_score"


def test_bear_market_blocks_new_buy() -> None:
    state = PortfolioState(
        config=PortfolioConfig(total_capital=300000, min_score=85),
        configured=True,
    )
    decision = decide_action(stock_id="2330", score=95, market_regime="BEAR", state=state)
    assert decision["action"] == "NO_TRADE"
    assert not decision["selected"]


def test_above_minimum_score_buys_in_bull_market() -> None:
    state = PortfolioState(
        config=PortfolioConfig(total_capital=300000, min_score=85),
        configured=True,
    )
    decision = decide_action(stock_id="2330", score=95, market_regime="BULL", state=state)
    assert decision["action"] == "BUY_NEXT_OPEN"
    assert decision["selected"]


def test_missing_portfolio_state_does_not_disable_signal_scan(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    signals, _ = build_signal_sheet(_synthetic_universe(), portfolio_state_path=missing)
    assert not signals.empty
    # Without known holdings we still expose the technical signal; only the
    # HOLD classification requires a supplied positions list.
    assert "action" in signals.columns
