from __future__ import annotations

"""Portfolio-aware decision engine for V1.6 live signals.

The engine does not place orders. It converts deterministic stock candidates into
BUY/HOLD/WATCH/NO_TRADE decisions using explicit capital and activity limits.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = Path("data/portfolio_state.json")


@dataclass(frozen=True)
class PortfolioConfig:
    total_capital: float = 0.0
    cash_available: float = 0.0
    max_positions: int = 5
    max_position_pct: float = 0.20
    target_position_pct: float = 0.15
    max_daily_trades: int = 1
    max_monthly_trades: int = 10
    min_score: float = 85.0
    min_order_amount: float = 1000.0
    bear_market_entry: bool = False


@dataclass
class PortfolioState:
    config: PortfolioConfig
    positions: dict[str, float] = field(default_factory=dict)
    trade_history: list[date] = field(default_factory=list)
    last_exit_dates: dict[str, date] = field(default_factory=dict)
    configured: bool = False
    source: str = "defaults"

    @property
    def position_count(self) -> int:
        return sum(1 for value in self.positions.values() if value > 0)

    @property
    def invested_value(self) -> float:
        return sum(max(value, 0.0) for value in self.positions.values())

    @property
    def remaining_slots(self) -> int:
        return max(self.config.max_positions - self.position_count, 0)

    def monthly_trades_used(self, as_of: date) -> int:
        return sum(
            1 for item in self.trade_history
            if item.year == as_of.year and item.month == as_of.month
        )


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None


def _normalise_config(raw: dict[str, Any]) -> PortfolioConfig:
    return PortfolioConfig(
        total_capital=float(raw.get("total_capital", 0.0)),
        cash_available=float(raw.get("cash_available", 0.0)),
        max_positions=max(int(raw.get("max_positions", 5)), 1),
        max_position_pct=min(max(float(raw.get("max_position_pct", 0.20)), 0.0), 1.0),
        target_position_pct=min(max(float(raw.get("target_position_pct", 0.15)), 0.0), 1.0),
        max_daily_trades=max(int(raw.get("max_daily_trades", 1)), 0),
        max_monthly_trades=max(int(raw.get("max_monthly_trades", 10)), 0),
        min_score=float(raw.get("min_score", 85.0)),
        min_order_amount=max(float(raw.get("min_order_amount", 1000.0)), 0.0),
        bear_market_entry=bool(raw.get("bear_market_entry", False)),
    )


def load_portfolio_state(path: str | Path = DEFAULT_STATE_PATH) -> PortfolioState:
    path = Path(path)
    if not path.exists():
        return PortfolioState(config=PortfolioConfig(), configured=False, source=str(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    config = _normalise_config(raw.get("config", {}))
    positions = {
        str(k): float(v)
        for k, v in raw.get("positions", {}).items()
        if float(v) > 0
    }
    trade_history = [
        parsed
        for parsed in (_parse_date(item) for item in raw.get("trade_history", []))
        if parsed is not None
    ]
    last_exit_dates = {
        str(k): parsed
        for k, v in raw.get("last_exit_dates", {}).items()
        if (parsed := _parse_date(v)) is not None
    }
    configured = config.total_capital > 0 and config.cash_available >= 0
    return PortfolioState(
        config=config,
        positions=positions,
        trade_history=trade_history,
        last_exit_dates=last_exit_dates,
        configured=configured,
        source=str(path),
    )


def _amount_floor(value: float, minimum: float) -> float:
    if value <= 0:
        return 0.0
    if minimum <= 0:
        return value
    return float(int(value / minimum) * minimum)


def allocate_candidate(
    *,
    stock_id: str,
    score: float,
    close: float,
    market_regime: str,
    signal_date: date,
    state: PortfolioState,
    trades_today: int,
    cash_available: float | None = None,
) -> dict[str, Any]:
    """Return a deterministic portfolio-aware action for one candidate."""
    cfg = state.config
    available_cash = cfg.cash_available if cash_available is None else max(cash_available, 0.0)
    result: dict[str, Any] = {
        "stock_id": stock_id,
        "action": "WATCH",
        "selected": False,
        "allocation_amount": 0.0,
        "allocation_pct": 0.0,
        "portfolio_reason": "not_selected",
        "trade_score": float(score),
        "remaining_slots": state.remaining_slots,
        "monthly_trades_used": state.monthly_trades_used(signal_date),
        "configured": state.configured,
    }

    if not state.configured:
        result["action"] = "NO_TRADE"
        result["portfolio_reason"] = "portfolio_state_not_configured"
        return result

    if stock_id in state.positions:
        result["action"] = "HOLD"
        result["portfolio_reason"] = "already_held"
        return result

    if score < cfg.min_score:
        result["portfolio_reason"] = "below_portfolio_min_score"
        return result

    if market_regime.upper() == "BEAR" and not cfg.bear_market_entry:
        result["action"] = "NO_TRADE"
        result["portfolio_reason"] = "bear_market_entry_block"
        return result

    if state.remaining_slots <= 0:
        result["portfolio_reason"] = "max_positions_reached"
        return result

    if trades_today >= cfg.max_daily_trades:
        result["portfolio_reason"] = "daily_trade_limit_reached"
        return result

    monthly_used = state.monthly_trades_used(signal_date)
    if monthly_used >= cfg.max_monthly_trades:
        result["portfolio_reason"] = "monthly_trade_limit_reached"
        return result

    max_amount = min(available_cash, cfg.total_capital * cfg.max_position_pct)
    target_amount = min(max_amount, cfg.total_capital * cfg.target_position_pct)
    allocation = _amount_floor(target_amount, cfg.min_order_amount)

    if close <= 0:
        result["portfolio_reason"] = "invalid_price"
        return result
    if allocation <= 0 or allocation > available_cash:
        result["portfolio_reason"] = "insufficient_cash_for_min_order"
        return result

    result.update(
        {
            "action": "BUY_NEXT_OPEN",
            "selected": True,
            "allocation_amount": allocation,
            "allocation_pct": allocation / cfg.total_capital if cfg.total_capital else 0.0,
            "portfolio_reason": "passed_portfolio_constraints",
        }
    )
    return result


def apply_portfolio_decisions(
    signals,
    state: PortfolioState,
    *,
    signal_date: date,
    max_candidates: int = 10,
):
    """Rank V1.5 candidates, then apply V1.6 portfolio constraints without forcing trades."""
    import pandas as pd

    result = signals.copy()
    if result.empty:
        result["selected"] = pd.Series(dtype=bool)
        result["action"] = pd.Series(dtype=str)
        result["allocation_amount"] = pd.Series(dtype=float)
        result["allocation_pct"] = pd.Series(dtype=float)
        result["portfolio_reason"] = pd.Series(dtype=str)
        result["trade_score"] = pd.Series(dtype=float)
        result["remaining_slots"] = pd.Series(dtype=int)
        result["monthly_trades_used"] = pd.Series(dtype=int)
        result["configured"] = pd.Series(dtype=bool)
        return result

    result = result.sort_values(
        ["candidate", "score", "rs20", "stock_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    candidate_idx = result.index[result["candidate"]].tolist()[:max_candidates]

    result["selected"] = False
    result["action"] = "WATCH"
    result["allocation_amount"] = 0.0
    result["allocation_pct"] = 0.0
    result["portfolio_reason"] = result["candidate"].map(
        lambda is_candidate: "not_in_top_candidate_set" if not bool(is_candidate) else "not_selected"
    )
    result["trade_score"] = result["score"].astype(float)
    result["remaining_slots"] = state.remaining_slots
    result["monthly_trades_used"] = state.monthly_trades_used(signal_date)
    result["configured"] = state.configured
    result["portfolio_cash_available"] = state.config.cash_available
    result["portfolio_total_capital"] = state.config.total_capital
    result["portfolio_position_count"] = state.position_count
    result["portfolio_max_positions"] = state.config.max_positions
    result["portfolio_max_daily_trades"] = state.config.max_daily_trades
    result["portfolio_max_monthly_trades"] = state.config.max_monthly_trades

    decisions: dict[int, dict[str, Any]] = {}
    trades_today = 0
    cash_remaining = state.config.cash_available
    for idx in candidate_idx:
        row = result.loc[idx]
        decision = allocate_candidate(
            stock_id=str(row["stock_id"]),
            score=float(row["score"]),
            close=float(row["close"]),
            market_regime=str(row["market_regime"]),
            signal_date=signal_date,
            state=state,
            trades_today=trades_today,
            cash_available=cash_remaining,
        )
        decisions[idx] = decision
        if decision["selected"]:
            trades_today += 1
            cash_remaining = max(
                cash_remaining - float(decision["allocation_amount"]), 0.0
            )

    for idx, decision in decisions.items():
        for key, value in decision.items():
            if key != "stock_id":
                result.at[idx, key] = value

    return result
