from __future__ import annotations

import numpy as np
import pandas as pd

from daily_signal import (
    MAX_POSITIONS,
    MIN_SCORE,
    PULLBACK_MAX,
    PULLBACK_MIN,
    RS_THRESHOLD,
    build_signal_sheet,
    is_v15_candidate,
    rank_and_select_signals,
)


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
    """One deterministic V1.5 candidate plus five weak/flat stocks."""
    n = 90
    weak = np.linspace(100.0, 100.5, n).tolist()

    # The final 20 sessions rise from 90 to a 110 peak, then pull back to
    # 105. This deliberately satisfies the frozen V1.5 pullback window while
    # keeping 20D/60D momentum and the MA20 trend positive.
    strong = list(np.linspace(70.0, 90.0, 70))
    strong.extend(np.linspace(90.0, 110.0, 10))
    strong.extend(np.linspace(110.0, 105.0, 10))

    return {
        "STRONG": _make_stock(strong),
        "WEAK1": _make_stock(weak),
        "WEAK2": _make_stock(weak),
        "WEAK3": _make_stock(weak),
        "WEAK4": _make_stock(weak),
        "WEAK5": _make_stock(weak),
    }


def test_v15_threshold_constants_are_frozen() -> None:
    assert MAX_POSITIONS == 3
    assert RS_THRESHOLD == 0.10
    assert PULLBACK_MIN == -0.07
    assert PULLBACK_MAX == -0.02
    assert MIN_SCORE == 70.0


def test_entry_candidate_rule_is_regime_independent() -> None:
    kwargs = {
        "rs20": 0.15,
        "drawdown_20d": -0.05,
        "score": 80.0,
        "trend_pass": True,
        "momentum_pass": True,
    }
    assert is_v15_candidate(**kwargs)


def test_daily_signal_returns_latest_date_and_expected_columns() -> None:
    signals, regime = build_signal_sheet(_synthetic_universe())

    assert not signals.empty
    assert signals["date"].nunique() == 1
    assert regime in {"bull", "neutral", "bear"}

    expected = {
        "date",
        "stock_id",
        "market_regime",
        "close",
        "rs20",
        "rs60",
        "drawdown_20d",
        "score",
        "trend_pass",
        "momentum_pass",
        "pullback_pass",
        "candidate",
        "selected",
        "action",
        "reason",
    }
    assert expected.issubset(signals.columns)


def test_candidate_respects_frozen_v15_rules() -> None:
    signals, _ = build_signal_sheet(_synthetic_universe())
    candidates = signals[signals["candidate"]]

    assert not candidates.empty
    assert (candidates["rs20"] > RS_THRESHOLD).all()
    assert (candidates["drawdown_20d"] >= PULLBACK_MIN).all()
    assert (candidates["drawdown_20d"] <= PULLBACK_MAX).all()
    assert (candidates["score"] >= MIN_SCORE).all()
    assert candidates["trend_pass"].all()
    assert candidates["momentum_pass"].all()


def test_selected_signals_are_ranked_and_capped() -> None:
    signals = pd.DataFrame(
        [
            {"stock_id": "A", "candidate": True, "rs20": 0.15, "score": 72.0},
            {"stock_id": "B", "candidate": True, "rs20": 0.30, "score": 71.0},
            {"stock_id": "C", "candidate": True, "rs20": 0.20, "score": 95.0},
            {"stock_id": "D", "candidate": True, "rs20": 0.30, "score": 80.0},
            {"stock_id": "E", "candidate": False, "rs20": 0.50, "score": 99.0},
        ]
    )

    ranked = rank_and_select_signals(signals)
    selected = ranked[ranked["selected"]]

    assert len(selected) == MAX_POSITIONS
    assert selected["stock_id"].tolist() == ["D", "B", "C"]
    assert selected["action"].eq("BUY_NEXT_OPEN").all()
    assert ranked.loc[~ranked["selected"], "action"].eq("WATCH").all()


def test_non_candidates_are_not_marked_for_entry() -> None:
    signals, _ = build_signal_sheet(_synthetic_universe())
    non_candidates = signals[~signals["candidate"]]

    assert (non_candidates["selected"] == False).all()
    assert (non_candidates["action"] == "WATCH").all()
    assert (non_candidates["reason"] == "does_not_meet_v1_5_rules").all()
