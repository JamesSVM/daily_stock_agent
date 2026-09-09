from __future__ import annotations

"""Portfolio-aware decision gates and budget allocation for V1.6."""

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Any

DEFAULT_STATE_PATH = Path("data/portfolio_state.json")

@dataclass(frozen=True)
class PortfolioConfig:
    total_capital: float = 0.0
    cash_available: float = 0.0
    max_positions: int = 999999
    max_position_pct: float = 1.0
    target_position_pct: float = 0.15
    max_daily_trades: int = 1
    max_monthly_trades: int = 10
    min_score: float = 85.0
    min_order_amount: float = 0.0
    bear_market_entry: bool = False

@dataclass
class PortfolioState:
    config: PortfolioConfig
    positions: dict[str, float] = field(default_factory=dict)
    configured: bool = False
    source: str = "defaults"

    @property
    def position_count(self) -> int:
        return len(self.positions)

def _normalise_config(raw: dict[str, Any]) -> PortfolioConfig:
    return PortfolioConfig(
        total_capital=max(float(raw.get("total_capital", 0.0)), 0.0),
        cash_available=max(float(raw.get("cash_available", raw.get("total_capital", 0.0))), 0.0),
        max_positions=max(int(raw.get("max_positions", 999999)), 0),
        max_position_pct=max(float(raw.get("max_position_pct", 1.0)), 0.0),
        target_position_pct=max(float(raw.get("target_position_pct", 0.15)), 0.0),
        max_daily_trades=max(int(raw.get("max_daily_trades", 1)), 0),
        max_monthly_trades=max(int(raw.get("max_monthly_trades", 10)), 0),
        min_score=float(raw.get("min_score", 85.0)),
        min_order_amount=max(float(raw.get("min_order_amount", 0.0)), 0.0),
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
        positions = {str(stock_id): float(value) for stock_id, value in raw_positions.items() if float(value) > 0}
    elif isinstance(raw_positions, list):
        positions = {str(stock_id): 1.0 for stock_id in raw_positions}
    else:
        positions = {}
    configured = config.total_capital > 0 or bool(positions)
    return PortfolioState(config=config, positions=positions, configured=configured, source=str(path))

def allocate_candidate(*, stock_id: str, score: float, close: float, market_regime: str, signal_date: date, state: PortfolioState, trades_today: int = 0, trades_this_month: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"stock_id": stock_id, "action": "WATCH", "selected": False, "trade_score": float(score), "portfolio_reason": "not_selected", "allocation_amount": 0.0, "allocation_shares": 0, "configured": state.configured, "total_capital": state.config.total_capital, "position_count": state.position_count}
    if not state.configured:
        result.update(action="NO_TRADE", portfolio_reason="portfolio_state_not_configured")
        return result
    if stock_id in state.positions:
        result.update(action="HOLD", portfolio_reason="already_held")
        return result
    if score < state.config.min_score:
        result["portfolio_reason"] = "below_minimum_score"
        return result
    if market_regime.upper() == "BEAR" and not state.config.bear_market_entry:
        result.update(action="NO_TRADE", portfolio_reason="bear_market_entry_block")
        return result
    if state.position_count >= state.config.max_positions:
        result["portfolio_reason"] = "max_positions_reached"
        return result
    if trades_today >= state.config.max_daily_trades:
        result["portfolio_reason"] = "daily_trade_limit_reached"
        return result
    if trades_this_month >= state.config.max_monthly_trades:
        result["portfolio_reason"] = "monthly_trade_limit_reached"
        return result
    if close <= 0:
        result["portfolio_reason"] = "invalid_price"
        return result
    target_amount = state.config.total_capital * state.config.target_position_pct
    max_amount = state.config.total_capital * state.config.max_position_pct
    allocation_amount = min(target_amount, max_amount, state.config.cash_available)
    if allocation_amount < state.config.min_order_amount or allocation_amount < close:
        result["portfolio_reason"] = "insufficient_cash"
        return result
    shares = int(allocation_amount // close)
    if shares <= 0:
        result["portfolio_reason"] = "insufficient_cash"
        return result
    allocation_amount = shares * close
    result.update(action="BUY_NEXT_OPEN", selected=True, portfolio_reason="allocated_within_budget", allocation_amount=float(allocation_amount), allocation_shares=shares)
    return result

def decide_action(*, stock_id: str, score: float, market_regime: str, state: PortfolioState) -> dict[str, Any]:
    return allocate_candidate(stock_id=stock_id, score=score, close=1.0, market_regime=market_regime, signal_date=date.today(), state=state, trades_today=0)

def apply_portfolio_decisions(signals, state: PortfolioState, *, signal_date, max_candidates: int = 10):
    import pandas as pd
    result = signals.copy()
    columns = {"selected": bool, "top_candidate": bool, "action": str, "trade_score": float, "portfolio_reason": str, "configured": bool, "total_capital": float, "position_count": int, "allocation_amount": float, "allocation_shares": int}
    if result.empty:
        for column, dtype in columns.items():
            result[column] = pd.Series(dtype=dtype)
        return result
    result = result.sort_values(["candidate", "score", "rs20", "stock_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    result["selected"] = False
    result["top_candidate"] = False
    result["action"] = "WATCH"
    result["trade_score"] = result["score"].astype(float)
    result["portfolio_reason"] = result["candidate"].map(lambda value: "not_in_top_candidate_set" if not bool(value) else "not_selected")
    result["configured"] = state.configured
    result["total_capital"] = state.config.total_capital
    result["position_count"] = state.position_count
    result["allocation_amount"] = 0.0
    result["allocation_shares"] = 0
    candidate_idx = result.index[result["candidate"]].tolist()
    for idx in candidate_idx[:max_candidates]:
        result.loc[idx, "top_candidate"] = True
    trades_today = 0
    for idx in candidate_idx:
        row = result.loc[idx]
        decision = allocate_candidate(stock_id=str(row["stock_id"]), score=float(row["score"]), close=float(row.get("close", 0.0)), market_regime=str(row["market_regime"]), signal_date=signal_date, state=state, trades_today=trades_today)
        for key, value in decision.items():
            if key != "stock_id":
                result.at[idx, key] = value
        if decision.get("selected"):
            trades_today += 1
    return result
