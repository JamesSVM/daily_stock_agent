from __future__ import annotations

import pandas as pd

from features.mean_reversion import add_mean_reversion_features, build_signal


def simulate_stock(
    df: pd.DataFrame,
    stock_id: str,
    hold_days: int = 10,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    min_score: float = 70.0,
) -> list[dict]:
    """Event-driven single-stock backtest.

    Signal is evaluated at the close of day T and the trade enters at the next
    trading day's open. Stop/target are checked using future daily high/low.
    This avoids using the same day's close to pretend we could fill earlier.
    """
    data = add_mean_reversion_features(df).copy()
    data["prev_rsi14"] = data["rsi14"].shift(1)
    data = data.dropna(subset=["ma60", "rsi14", "atr14"])

    trades: list[dict] = []
    i = 0

    while i < len(data) - 1:
        row = data.iloc[i]
        signal = build_signal(row)

        if not signal["buy_zone"] or signal["score"] < min_score:
            i += 1
            continue

        entry_i = i + 1
        entry_row = data.iloc[entry_i]
        entry_price = float(entry_row["Open"])
        stop = entry_price * (1 - stop_loss)
        target = entry_price * (1 + take_profit)

        exit_i = min(entry_i + hold_days - 1, len(data) - 1)
        exit_price = float(data.iloc[exit_i]["Close"])
        exit_reason = "time_exit"

        for j in range(entry_i, exit_i + 1):
            future = data.iloc[j]
            low = float(future["Low"])
            high = float(future["High"])

            # Conservative assumption when both levels are touched intraday:
            # stop is hit first.
            if low <= stop:
                exit_i = j
                exit_price = stop
                exit_reason = "stop_loss"
                break
            if high >= target:
                exit_i = j
                exit_price = target
                exit_reason = "take_profit"
                break

        ret = exit_price / entry_price - 1
        trades.append(
            {
                "stock_id": stock_id,
                "signal_date": data.index[i] if data.index.name else row.get("Date", i),
                "entry_date": data.index[entry_i] if data.index.name else entry_row.get("Date", entry_i),
                "exit_date": data.index[exit_i] if data.index.name else data.iloc[exit_i].get("Date", exit_i),
                "holding_days": exit_i - entry_i + 1,
                "score": signal["score"],
                "oversold_score": signal["oversold_score"],
                "reversal_score": signal["reversal_score"],
                "entry": entry_price,
                "exit": exit_price,
                "return": ret,
                "exit_reason": exit_reason,
            }
        )

        # Do not generate overlapping positions on the same stock.
        i = exit_i + 1

    return trades


def summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "trades_per_month": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
        }

    df = pd.DataFrame(trades)
    returns = df["return"].astype(float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    dates = pd.to_datetime(df["entry_date"], errors="coerce")
    months = max(dates.dt.to_period("M").nunique(), 1)
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    return {
        "trades": int(len(df)),
        "trades_per_month": round(len(df) / months, 2),
        "win_rate": round(float((returns > 0).mean() * 100), 2),
        "avg_return": round(float(returns.mean() * 100), 2),
        "expectancy": round(float(returns.mean() * 100), 2),
        "profit_factor": round(float(gross_profit / gross_loss), 2) if gross_loss else 0.0,
    }
