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
    """One strong pullback candidate plus five weak/flat stocks."""
    n = 90
    weak = np.linspace(100.0, 100.5, n).tolist()

    # Strong stock: long trend, positive 60D/20D momentum, then a 5% pullback
    # from the 20D high while remaining above MA20 and MA60.
    strong = list(np.linspace(70.0, 100.0, 70))
    strong.extend(np.linspace(100.0, 108.0, 15))
    strong.extend([106.0, 105.0, 103.5, 102.5, 102.0])

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
    signals, _ = build_signal_sheet(_synthetic_universe())
    selected = signals[signals["selected"]]

    assert len(selected) <= MAX_POSITIONS
    assert (selected["action"] == "BUY_NEXT_OPEN").all()
    assert signals.loc[~signals["selected"], "action"].eq("WATCH").all()

    selected_scores = selected["score"].tolist()
    assert selected_scores == sorted(selected_scores, reverse=True)


def test_non_candidates_are_not_marked_for_entry() -> None:
    signals, _ = build_signal_sheet(_synthetic_universe())
    non_candidates = signals[~signals["candidate"]]

    assert (non_candidates["selected"] == False).all()
    assert (non_candidates["action"] == "WATCH").all()
    assert (non_candidates["reason"] == "does_not_meet_v1_5_rules").all()
