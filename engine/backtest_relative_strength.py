from __future__ import annotations

import pandas as pd

from features.relative_strength import (
    add_relative_strength_features,
    build_relative_strength_signal,
)


def simulate_stock_relative_strength(
    df: pd.DataFrame,
    stock_id: str,
    market: pd.DataFrame,
    hold_days: int = 10,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    min_score: float = 70.0,
    allowed_regimes: set[str] | None = None,
) -> list[dict]:
    """
    V1.3 Relative Strength + Momentum backtest.

    Signal:
        Relative Strength
        + Trend
        + Momentum
        + Pullback

    Entry:
        Signal on T close
        Entry on T+1 open

    Exit:
        Stop loss
        Take profit
        Time exit
    """

    data = add_relative_strength_features(
        df,
        market,
    ).copy()

    data = data.dropna(
        subset=[
            "ma20",
            "ma60",
            "rs20",
            "rs60",
            "return_20d",
            "return_60d",
        ]
    )

    trades: list[dict] = []

    i = 0

    while i < len(data) - 1:

        signal_date = data.index[i]
        row = data.iloc[i]

        # ----------------------------------------------
        # Market regime
        # ----------------------------------------------

        market_regime = "neutral"

        if signal_date in market.index:
            market_regime = str(
                market.loc[
                    signal_date,
                    "regime",
                ]
            )

        if (
            allowed_regimes is not None
            and market_regime not in allowed_regimes
        ):
            i += 1
            continue

        # ----------------------------------------------
        # Signal
        # ----------------------------------------------

        signal = build_relative_strength_signal(
            row,
        )

        if (
            not signal["signal"]
            or signal["score"] < min_score
        ):
            i += 1
            continue

        # ----------------------------------------------
        # Entry
        # ----------------------------------------------

        entry_i = i + 1

        if entry_i >= len(data):
            break

        entry_row = data.iloc[entry_i]

        entry_date = data.index[entry_i]

        entry_price = float(
            entry_row["Open"]
        )

        stop = (
            entry_price
            * (1 - stop_loss)
        )

        target = (
            entry_price
            * (1 + take_profit)
        )

        # ----------------------------------------------
        # Exit
        # ----------------------------------------------

        exit_i = min(
            entry_i + hold_days - 1,
            len(data) - 1,
        )

        exit_price = float(
            data.iloc[exit_i]["Close"]
        )

        exit_reason = "time_exit"

        for j in range(
            entry_i,
            exit_i + 1,
        ):

            future = data.iloc[j]

            low = float(
                future["Low"]
            )

            high = float(
                future["High"]
            )

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

        exit_date = data.index[exit_i]

        ret = (
            exit_price
            / entry_price
            - 1.0
        )

        # ----------------------------------------------
        # Market return / Alpha
        # ----------------------------------------------

        market_return = 0.0

        market_after = market.loc[
            market.index <= exit_date,
            "close",
        ]

        market_before = market.loc[
            market.index >= entry_date,
            "close",
        ]

        if (
            not market_after.empty
            and not market_before.empty
        ):

            market_start = float(
                market_before.iloc[0]
            )

            market_end = float(
                market_after.iloc[-1]
            )

            if market_start != 0:
                market_return = (
                    market_end
                    / market_start
                    - 1.0
                )

        alpha = (
            ret
            - market_return
        )

        trades.append(
            {
                "stock_id": stock_id,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_days": (
                    exit_i
                    - entry_i
                    + 1
                ),
                "score": signal["score"],
                "rs20": signal["rs20"],
                "drawdown_20d": signal["drawdown_20d"],
                "rs60": signal["rs60"],
                "rs20_score": signal["rs20_score"],
                "rs60_score": signal["rs60_score"],
                "trend_pass": signal["trend_pass"],
                "momentum_pass": signal["momentum_pass"],
                "pullback_pass": signal["pullback_pass"],
                "entry": entry_price,
                "exit": exit_price,
                "return": ret,
                "market_return": market_return,
                "alpha": alpha,
                "market_regime": market_regime,
                "exit_reason": exit_reason,
            }
        )

        # No overlapping position.
        i = exit_i + 1

    return trades
