from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable


@dataclass(frozen=True)
class TradeCandidate:
    stock_id: str
    date: object
    score: float
    entry: float
    stop: float
    target: float
    sector: str | None = None

    @property
    def risk_reward(self) -> float:
        risk = self.entry - self.stop
        if risk <= 0:
            return 0.0
        return (self.target - self.entry) / risk


def passes_trade_filter(candidate: TradeCandidate, min_score: float = 70.0, min_rr: float = 2.0) -> bool:
    return candidate.score >= min_score and candidate.risk_reward >= min_rr


def select_candidates(
    candidates: Iterable[TradeCandidate],
    max_positions: int = 10,
    min_score: float = 70.0,
    min_rr: float = 2.0,
    sector_max_positions: int = 2,
) -> list[TradeCandidate]:
    """Rank candidates while limiting concentration. No artificial quota is imposed."""
    selected: list[TradeCandidate] = []
    sector_counts: dict[str, int] = {}

    ordered = sorted(candidates, key=lambda x: x.score, reverse=True)

    for candidate in ordered:
        if len(selected) >= max_positions:
            break
        if not passes_trade_filter(candidate, min_score=min_score, min_rr=min_rr):
            continue

        sector = candidate.sector or "UNKNOWN"
        if sector_counts.get(sector, 0) >= sector_max_positions:
            continue

        selected.append(candidate)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return selected


def in_cooldown(
    stock_id: str,
    entry_date,
    previous_exit_dates: dict[str, object],
    cooldown_days: int = 10,
) -> bool:
    """Block re-entry for N calendar days after the previous exit."""
    last_exit = previous_exit_dates.get(stock_id)
    if last_exit is None:
        return False
    return entry_date < last_exit + timedelta(days=cooldown_days)
