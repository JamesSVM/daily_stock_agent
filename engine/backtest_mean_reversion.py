from __future__ import annotations

import pandas as pd

from features.mean_reversion import (
    add_mean_reversion_features,
    build_signal,
)
from engine.market_regime import (
    get_market_regime,
    get_market_return,
)


def simulate_stock(
    df: pd.DataFrame,
    stock_id: str,
    hold_days: int = 10,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    min_score: float = 70.0,
    market=None,
    allowed_regimes=None,
) -> list[dict]:
    """Event-driven single-stock backtest.

    Signal is evaluated at the close of day T and the trade enters
    at the next trading day's open.

    V1.3 additions:
    - market regime
    - market return
    - alpha
    - 20-day relative strength

    Alpha is measured as:

        stock return - market return

    over the signal-date to exit-date window.
    """

    data = df.copy()

    # --------------------------------------------------
    # Normalize Date index
    # --------------------------------------------------

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce",
        )

        data = data.dropna(
            subset=["Date"],
        )

        data = data.set_index("Date")

    elif not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        data.index = pd.to_datetime(
            data.index,
            errors="coerce",
        )

        data = data[
            ~data.index.isna()
        ]

    data = data.sort_index()

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    data = add_mean_reversion_features(
        data
    ).copy()

    data["prev_rsi14"] = (
        data["rsi14"].shift(1)
    )

    data = data.dropna(
        subset=[
            "ma60",
            "rsi14",
            "atr14",
        ]
    )

    trades: list[dict] = []

    i = 0

    while i < len(data) - 1:

        row = data.iloc[i]
        signal_date = data.index[i]

        # --------------------------------------------------
        # Market regime filter
        # --------------------------------------------------

        market_regime = "neutral"

        if market is not None:

            market_regime = get_market_regime(
                market,
                signal_date,
            )

            if (
                allowed_regimes is not None
                and market_regime not in allowed_regimes
            ):
                i += 1
                continue

        # --------------------------------------------------
        # Signal
        # --------------------------------------------------

        signal = build_signal(row)

        if (
            not signal["buy_zone"]
            or signal["score"] < min_score
        ):
            i += 1
            continue

        # --------------------------------------------------
        # Entry
        # --------------------------------------------------

        entry_i = i + 1

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

        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

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

            # Conservative assumption:
            # stop is hit first if both levels
            # are touched on the same day.

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

        # --------------------------------------------------
        # Returns
        # --------------------------------------------------

        ret = (
            exit_price
            / entry_price
            - 1
        )

        # --------------------------------------------------
        # Market / Alpha
        # --------------------------------------------------

        market_return = 0.0
        alpha = 0.0
        relative_strength_20d = 0.0

        if market is not None:

            market_return = get_market_return(
                market,
                signal_date,
                exit_date,
            )

            alpha = (
                ret
                - market_return
            )

            if signal_date in market.index:

                stock_close_now = float(
                    row["Close"]
                )

                market_close_now = float(
                    market.loc[
                        signal_date,
                        "close",
                    ]
                )

                # 20D relative strength
                # based on stock vs market.

                stock_20d_return = 0.0

                if i >= 20:

                    past_close = float(
                        data.iloc[i - 20]["Close"]
                    )

                    if past_close != 0:

                        stock_20d_return = (
                            stock_close_now
                            / past_close
                            - 1
                        )

                market_20d_return = float(
                    market.loc[
                        signal_date,
                        "return_20d",
                    ]
                )

                if pd.notna(
                    market_20d_return
                ):

                    relative_strength_20d = (
                        stock_20d_return
                        - market_20d_return
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

                "oversold_score": (
                    signal["oversold_score"]
                ),

                "reversal_score": (
                    signal["reversal_score"]
                ),

                "entry": entry_price,

                "exit": exit_price,

                "return": ret,

                "market_return": (
                    market_return
                ),

                "alpha": alpha,

                "relative_strength_20d": (
                    relative_strength_20d
                ),

                "market_regime": (
                    market_regime
                ),

                "exit_reason": (
                    exit_reason
                ),
            }
        )

        # No overlapping positions.
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
