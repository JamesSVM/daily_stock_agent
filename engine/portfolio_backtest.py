from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Configurable Taiwan cash-equity trading cost assumptions."""

    buy_commission: float = 0.001425
    sell_commission: float = 0.001425
    sell_tax: float = 0.003
    buy_slippage: float = 0.0005
    sell_slippage: float = 0.0005


def _buy_cash_required(price: float, quantity: float, costs: CostModel) -> float:
    executed = price * (1.0 + costs.buy_slippage)
    notional = executed * quantity
    return notional * (1.0 + costs.buy_commission)


def _sell_cash_proceeds(price: float, quantity: float, costs: CostModel) -> float:
    executed = price * (1.0 - costs.sell_slippage)
    notional = executed * quantity
    return notional * (1.0 - costs.sell_commission - costs.sell_tax)


def simulate_portfolio(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    allocation_pct: float = 0.10,
    max_positions: int = 10,
    costs: CostModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply capital constraints to the existing single-stock trade signals."""
    costs = costs or CostModel()

    if trades.empty:
        return (
            pd.DataFrame(
                columns=["date", "cash", "positions", "exposure", "exposure_pct", "equity"]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            _empty_metrics(initial_capital, allocation_pct, max_positions, costs),
        )

    df = trades.copy()
    required = {"stock_id", "entry_date", "exit_date", "entry", "exit", "score", "exit_reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    for column in ("entry_date", "exit_date"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in ("entry", "exit", "score"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["stock_id", "entry_date", "exit_date", "entry", "exit", "score"]
    ).copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df = df.sort_values(
        ["entry_date", "score", "stock_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    df["trade_id"] = np.arange(len(df), dtype=int)

    price_frame = prices.copy()
    if price_frame.empty:
        raise ValueError("prices cannot be empty for portfolio simulation")
    price_frame["stock_id"] = price_frame["stock_id"].astype(str)
    price_frame["date"] = pd.to_datetime(price_frame["date"], errors="coerce")
    price_frame["close"] = pd.to_numeric(price_frame["close"], errors="coerce")
    price_frame = price_frame.dropna(subset=["stock_id", "date", "close"])
    close_lookup = price_frame.set_index(["date", "stock_id"])["close"]

    start_date = df["entry_date"].min()
    end_date = df["exit_date"].max()
    dates = pd.date_range(start_date, end_date, freq="B")
    entries_by_date = {
        date: group.to_dict("records")
        for date, group in df.groupby("entry_date", sort=True)
    }

    cash = float(initial_capital)
    positions: dict[int, dict[str, Any]] = {}
    equity_rows: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def close_positions_for_date(date: pd.Timestamp) -> None:
        nonlocal cash

        exit_ids = [
            trade_id
            for trade_id, position in positions.items()
            if position["exit_date"] == date
        ]

        for trade_id in exit_ids:
            position = positions.pop(trade_id)
            proceeds = _sell_cash_proceeds(
                position["exit_price"], position["quantity"], costs
            )
            cash += proceeds
            entry_cash = position["entry_cash"]
            pnl = proceeds - entry_cash
            fills.append(
                {
                    "trade_id": trade_id,
                    "stock_id": position["stock_id"],
                    "entry_date": position["entry_date"],
                    "exit_date": date,
                    "entry_price": position["entry_price"],
                    "exit_price": position["exit_price"],
                    "quantity": position["quantity"],
                    "score": position["score"],
                    "exit_reason": position["exit_reason"],
                    "gross_return_pct": (
                        position["exit_price"] / position["entry_price"] - 1.0
                    ) * 100.0,
                    "net_pnl": pnl,
                    "net_return_pct": pnl / entry_cash * 100.0 if entry_cash else 0.0,
                }
            )

    for date in dates:
        close_positions_for_date(date)

        candidates = entries_by_date.get(date, [])
        available_slots = max(max_positions - len(positions), 0)
        selected = candidates[:available_slots]
        skipped = candidates[available_slots:]

        for record in skipped:
            rejected.append(
                {
                    "trade_id": int(record["trade_id"]),
                    "entry_date": date,
                    "stock_id": str(record["stock_id"]),
                    "score": float(record["score"]),
                    "reason": "max_positions",
                }
            )

        for record in selected:
            stock_id = str(record["stock_id"])
            if any(position["stock_id"] == stock_id for position in positions.values()):
                rejected.append(
                    {
                        "trade_id": int(record["trade_id"]),
                        "entry_date": date,
                        "stock_id": stock_id,
                        "score": float(record["score"]),
                        "reason": "already_holding",
                    }
                )
                continue

            target_cash = min(cash, initial_capital * allocation_pct)
            unit_cash = (
                float(record["entry"])
                * (1.0 + costs.buy_slippage)
                * (1.0 + costs.buy_commission)
            )
            quantity = target_cash / unit_cash if unit_cash > 0 else 0.0
            if quantity <= 0:
                rejected.append(
                    {
                        "trade_id": int(record["trade_id"]),
                        "entry_date": date,
                        "stock_id": stock_id,
                        "score": float(record["score"]),
                        "reason": "zero_quantity",
                    }
                )
                continue

            entry_cash = _buy_cash_required(float(record["entry"]), quantity, costs)
            cash -= entry_cash
            positions[int(record["trade_id"])] = {
                "trade_id": int(record["trade_id"]),
                "stock_id": stock_id,
                "entry_date": date,
                "exit_date": record["exit_date"],
                "entry_price": float(record["entry"]),
                "exit_price": float(record["exit"]),
                "quantity": float(quantity),
                "entry_cash": float(entry_cash),
                "score": float(record["score"]),
                "exit_reason": str(record["exit_reason"]),
            }

        close_positions_for_date(date)

        exposure = 0.0
        equity = cash
        for position in positions.values():
            close_price = close_lookup.get((date, position["stock_id"]), np.nan)
            if pd.isna(close_price):
                close_price = position["entry_price"]
            close_price = float(close_price)
            exposure += close_price * position["quantity"]
            equity += _sell_cash_proceeds(close_price, position["quantity"], costs)

        equity_rows.append(
            {
                "date": date,
                "cash": cash,
                "positions": len(positions),
                "exposure": exposure,
                "exposure_pct": exposure / equity * 100.0 if equity > 0 else 0.0,
                "equity": equity,
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    fills_df = pd.DataFrame(fills)
    rejected_df = pd.DataFrame(rejected)
    metrics = _portfolio_metrics(
        equity_curve,
        fills_df,
        rejected_df,
        initial_capital,
        allocation_pct,
        max_positions,
        costs,
    )
    return equity_curve, fills_df, rejected_df, metrics


def _empty_metrics(
    initial_capital: float,
    allocation_pct: float,
    max_positions: int,
    costs: CostModel,
) -> dict[str, Any]:
    return {
        "initial_capital": initial_capital,
        "final_equity": initial_capital,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "calmar": 0.0,
        "avg_exposure_pct": 0.0,
        "max_exposure_pct": 0.0,
        "trades_executed": 0,
        "trades_rejected": 0,
        "win_rate_pct": 0.0,
        "avg_net_return_pct": 0.0,
        "profit_factor": 0.0,
        "allocation_pct": allocation_pct * 100.0,
        "max_positions": max_positions,
        "buy_commission_pct": costs.buy_commission * 100.0,
        "sell_commission_pct": costs.sell_commission * 100.0,
        "sell_tax_pct": costs.sell_tax * 100.0,
        "buy_slippage_pct": costs.buy_slippage * 100.0,
        "sell_slippage_pct": costs.sell_slippage * 100.0,
    }


def _portfolio_metrics(
    equity_curve: pd.DataFrame,
    fills: pd.DataFrame,
    rejected: pd.DataFrame,
    initial_capital: float,
    allocation_pct: float,
    max_positions: int,
    costs: CostModel,
) -> dict[str, Any]:
    if equity_curve.empty:
        return _empty_metrics(initial_capital, allocation_pct, max_positions, costs)

    equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())

    final_equity = float(equity.iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    elapsed_days = max(
        (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days,
        1,
    )
    cagr = (final_equity / initial_capital) ** (365.25 / elapsed_days) - 1.0

    if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0:
        sharpe = float(
            daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    result: dict[str, Any] = {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "avg_exposure_pct": round(float(equity_curve["exposure_pct"].mean()), 2),
        "max_exposure_pct": round(float(equity_curve["exposure_pct"].max()), 2),
        "trades_executed": int(len(fills)),
        "trades_rejected": int(len(rejected)),
        "allocation_pct": allocation_pct * 100.0,
        "max_positions": max_positions,
        "buy_commission_pct": costs.buy_commission * 100.0,
        "sell_commission_pct": costs.sell_commission * 100.0,
        "sell_tax_pct": costs.sell_tax * 100.0,
        "buy_slippage_pct": costs.buy_slippage * 100.0,
        "sell_slippage_pct": costs.sell_slippage * 100.0,
    }

    if not fills.empty:
        net_returns = pd.to_numeric(fills["net_return_pct"], errors="coerce").dropna()
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        result.update(
            {
                "win_rate_pct": round(float((net_returns > 0).mean() * 100.0), 2),
                "avg_net_return_pct": round(float(net_returns.mean()), 2),
                "profit_factor": round(gross_profit / gross_loss if gross_loss > 0 else 0.0, 2),
                "avg_holding_days": round(
                    float(
                        (
                            pd.to_datetime(fills["exit_date"])
                            - pd.to_datetime(fills["entry_date"])
                        ).dt.days.add(1).mean()
                    ),
                    2,
                ),
            }
        )

    return result
