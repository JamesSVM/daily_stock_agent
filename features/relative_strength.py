from __future__ import annotations

import pandas as pd


def add_relative_strength_features(
    df: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add stock-vs-market relative strength and momentum features.

    Expected stock columns:
        Close

    Expected market columns:
        close
    """

    data = df.copy()

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce",
        )
        data = data.dropna(subset=["Date"]).set_index("Date")

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(
            data.index,
            errors="coerce",
        )
        data = data[~data.index.isna()]

    data = data.sort_index()

    if "Close" not in data.columns:
        raise ValueError("Stock data must contain Close column.")

    market_data = market.copy()

    if not isinstance(
        market_data.index,
        pd.DatetimeIndex,
    ):
        market_data.index = pd.to_datetime(
            market_data.index,
            errors="coerce",
        )
        market_data = market_data[~market_data.index.isna()]

    market_data = market_data.sort_index()

    if "close" not in market_data.columns:
        raise ValueError("Market data must contain close column.")

    # --------------------------------------------------
    # Stock returns
    # --------------------------------------------------

    data["return_20d"] = (
        data["Close"]
        .pct_change(20)
    )

    data["return_60d"] = (
        data["Close"]
        .pct_change(60)
    )

    # --------------------------------------------------
    # Market returns
    # --------------------------------------------------

    market_data["market_return_20d"] = (
        market_data["close"]
        .pct_change(20)
    )

    market_data["market_return_60d"] = (
        market_data["close"]
        .pct_change(60)
    )

    # --------------------------------------------------
    # Align by actual calendar date
    # --------------------------------------------------

    data = data.join(
        market_data[
            [
                "market_return_20d",
                "market_return_60d",
            ]
        ],
        how="left",
    )

    # --------------------------------------------------
    # Relative strength
    # --------------------------------------------------

    data["rs20"] = (
        data["return_20d"]
        - data["market_return_20d"]
    )

    data["rs60"] = (
        data["return_60d"]
        - data["market_return_60d"]
    )

    # --------------------------------------------------
    # Trend
    # --------------------------------------------------

    data["ma20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["ma60"] = (
        data["Close"]
        .rolling(60)
        .mean()
    )

    data["ma20_slope"] = (
        data["ma20"]
        - data["ma20"].shift(5)
    ) / data["ma20"].shift(5)

    data["trend_pass"] = (
        (data["Close"] > data["ma20"])
        & (data["ma20"] > data["ma60"])
        & (data["ma20_slope"] > 0)
    )

    # --------------------------------------------------
    # Momentum
    # --------------------------------------------------

    data["momentum_pass"] = (
        (data["return_20d"] > 0)
        & (data["return_60d"] > 0)
    )

    # --------------------------------------------------
    # Pullback
    # --------------------------------------------------

    data["high_20d"] = (
        data["Close"]
        .rolling(20)
        .max()
    )

    data["drawdown_20d"] = (
        data["Close"] / data["high_20d"] - 1
    )

    # Pullback means:
    # still relatively strong, but slightly below
    # the recent 20D high.
    data["pullback_pass"] = (
        (data["drawdown_20d"] <= -0.02)
        & (data["drawdown_20d"] >= -0.10)
        & (data["rs20"] > 0)
    )

    return data


def build_relative_strength_signal(
    row: pd.Series,
) -> dict:
    """
    Build a transparent 0-100 alpha-oriented score.
    """

    # --------------------------------------------------
    # Relative strength score
    # --------------------------------------------------

    rs20 = float(row.get("rs20", 0.0))
    rs60 = float(row.get("rs60", 0.0))

    rs20_score = 50.0 + rs20 * 500.0
    rs60_score = 50.0 + rs60 * 300.0

    rs20_score = max(0.0, min(100.0, rs20_score))
    rs60_score = max(0.0, min(100.0, rs60_score))

    # --------------------------------------------------
    # Trend
    # --------------------------------------------------

    trend_score = 100.0 if bool(
        row.get("trend_pass", False)
    ) else 0.0

    # --------------------------------------------------
    # Momentum
    # --------------------------------------------------

    momentum_score = 100.0 if bool(
        row.get("momentum_pass", False)
    ) else 0.0

    # --------------------------------------------------
    # Pullback
    # --------------------------------------------------

    pullback_score = 100.0 if bool(
        row.get("pullback_pass", False)
    ) else 0.0

    # --------------------------------------------------
    # Weighted score
    # --------------------------------------------------

    score = (
        rs20_score * 0.30
        + rs60_score * 0.25
        + trend_score * 0.20
        + momentum_score * 0.15
        + pullback_score * 0.10
    )

    signal = (
        score >= 70.0
        and rs20 > 0
        and rs60 > 0
        and bool(row.get("trend_pass", False))
        and bool(row.get("momentum_pass", False))
        and bool(row.get("pullback_pass", False))
    )

    return {
        "signal": signal,
        "score": round(score, 2),
        "rs20": rs20,
        "drawdown_20d": float(
            row.get("drawdown_20d", 0.0)
        ),
        "rs60": rs60,
        "rs20_score": round(rs20_score, 2),
        "rs60_score": round(rs60_score, 2),
        "trend_pass": bool(
            row.get("trend_pass", False)
        ),
        "momentum_pass": bool(
            row.get("momentum_pass", False)
        ),
        "pullback_pass": bool(
            row.get("pullback_pass", False)
        ),
    }
