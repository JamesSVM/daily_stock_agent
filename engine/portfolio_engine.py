from __future__ import annotations

"""Lightweight portfolio-aware decision gates for V1.6.

V1.6 intentionally does not model cash availability, position sizing, or trade
budgets yet. The portfolio layer only applies the explicit rules chosen for the
live system: minimum score, BEAR-market entry block, and HOLD for existing
positions. Total capital is metadata only and never drives a BUY decision.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

DEFAULT_STATE_PATH = Path("data/portfolio_state.json")


@dataclass(frozen=True)
class PortfolioConfig:
    total_capital: float = 0.0
    min_score: float = 85.0
    bear_market_entry: bool = False


@dataclass
class PortfolioState:
    config: PortfolioConfig
    positions: set[str] = field(default_factory=set)
    configured: bool = False
    source: str = "defaults"

    @property
    def position_count(self) -> int:
        return len(self.positions)


def _normalise_config(raw: dict[str, Any]) -> PortfolioConfig:
    return PortfolioConfig(
        total_capital=max(float(raw.get("total_capital", 0.0)), 0.0),
        min_score=float(raw.get("min_score", 85.0)),
        bear_market_entry=bool(raw.get("bear_market_entry", False)),
    )


def load_portfolio_state(path: str | Path = DEFAULT_STATE_PATH) -> PortfolioState:
    path = Path(path)
    if not path.exists():
        return PortfolioState(config=PortfolioConfig(), configured=False, source=str(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _normalise_config(raw.get("config", {}))
    raw_positions = raw.get("positions", {})
    if isinstance(raw_positions, dict):
        positions = {str(stock_id) for stock_id, value in raw_positions.items() if float(value) > 0}
    elif isinstance(raw_positions, list):
        positions = {str(stock_id) for stock_id in raw_positions}
    else:
        positions = set()

    configured = config.total_capital > 0 or bool(positions)
    return PortfolioState(config=config, positions=positions, configured=configured, source=str(path))


def decide_action(
    *,
    stock_id: str,
    score: float,
    market_regime: str,
    state: PortfolioState,
) -> dict[str, Any]:
    """Return BUY/HOLD/WATCH/NO_TRADE using only the selected V1.6 rules."""
    result: dict[str, Any] = {
        "stock_id": stock_id,
        "action": "WATCH",
        "selected": False,
        "trade_score": float(score),
        "portfolio_reason": "not_selected",
        "configured": state.configured,
        "total_capital": state.config.total_capital,
        "position_count": state.position_count,
    }

    if stock_id in state.positions:
        result.update(action="HOLD", portfolio_reason="already_held")
        return result

    if score < state.config.min_score:
        result["portfolio_reason"] = "below_minimum_score"
        return result

    if market_regime.upper() == "BEAR" and not state.config.bear_market_entry:
        result.update(action="NO_TRADE", portfolio_reason="bear_market_entry_block")
        return result

    result.update(
        action="BUY_NEXT_OPEN",
        selected=True,
        portfolio_reason="passed_v1_6_entry_gates",
    )
    return result


def apply_portfolio_decisions(
    signals,
    state: PortfolioState,
    *,
    signal_date,
    max_candidates: int = 10,
):
    """Apply V1.6 gates to every V1.5 candidate.

    ``max_candidates`` is a report/display limit only. It does not restrict
    which candidates receive portfolio decisions or become BUYs.
    There is no cash check, position sizing, daily trade quota, monthly trade
    quota, or forced Top-N BUY quota in this version.
    """
    import pandas as pd

    result = signals.copy()
    if result.empty:
        for column, dtype in {
            "selected": bool,
            "top_candidate": bool,
            "action": str,
            "trade_score": float,
            "portfolio_reason": str,
            "configured": bool,
            "total_capital": float,
            "position_count": int,
        }.items():
            result[column] = pd.Series(dtype=dtype)
        return result

    result = result.sort_values(
        ["candidate", "score", "rs20", "stock_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    result["selected"] = False
    result["top_candidate"] = False
    result["action"] = "WATCH"
    result["trade_score"] = result["score"].astype(float)
    result["portfolio_reason"] = result["candidate"].map(
        lambda value: "not_candidate" if not bool(value) else "not_selected"
    )
    result["configured"] = state.configured
    result["total_capital"] = state.config.total_capital
    result["position_count"] = state.position_count

    candidate_idx = result.index[result["candidate"]].tolist()
    top_candidate_idx = candidate_idx[:max_candidates]
    if top_candidate_idx:
        result.loc[top_candidate_idx, "top_candidate"] = True

    # Portfolio decisions are evaluated against the full V1.5 candidate pool.
    # The Top 10 marker above is display-only and never controls BUY selection.
    for idx in candidate_idx:
        row = result.loc[idx]
        decision = decide_action(
            stock_id=str(row["stock_id"]),
            score=float(row["score"]),
            market_regime=str(row["market_regime"]),
            state=state,
        )
        for key, value in decision.items():
            if key != "stock_id":
                result.at[idx, key] = value

    return result
