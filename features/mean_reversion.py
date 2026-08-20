import numpy as np
import pandas as pd
import ta


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features used by the low-buy / mean-reversion strategy."""
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    df["rsi14"] = ta.momentum.rsi(close, window=14)
    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14)

    df["ret5"] = close.pct_change(5)
    df["ret20"] = close.pct_change(20)

    df["distance_ma20"] = close / df["ma20"] - 1.0
    df["distance_ma60"] = close / df["ma60"] - 1.0

    df["vol5"] = volume.rolling(5).mean()
    df["vol20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["vol20"]

    df["high60"] = close.rolling(60).max()
    df["drawdown60"] = close / df["high60"] - 1.0

    # Reversal confirmation: current close starts reclaiming the short-term range.
    df["reversal_confirmed"] = (
        (close > close.shift(1))
        & (close > df["ma5"])
        & (volume >= df["vol5"] * 0.9)
    )

    return df


def oversold_score(latest: pd.Series) -> float:
    """Score oversold conditions with a preferred zone rather than 'more down = better'."""
    score = 0.0

    rsi = latest.get("rsi14", np.nan)
    ret5 = latest.get("ret5", np.nan)
    distance_ma20 = latest.get("distance_ma20", np.nan)
    drawdown60 = latest.get("drawdown60", np.nan)

    if pd.notna(rsi):
        if 25 <= rsi <= 35:
            score += 30
        elif 20 <= rsi < 25 or 35 < rsi <= 40:
            score += 20
        elif rsi < 20:
            score += 5

    if pd.notna(ret5):
        if -0.15 <= ret5 <= -0.07:
            score += 25
        elif -0.20 <= ret5 < -0.15 or -0.07 < ret5 <= -0.05:
            score += 15

    if pd.notna(distance_ma20):
        if -0.15 <= distance_ma20 <= -0.07:
            score += 25
        elif -0.20 <= distance_ma20 < -0.15 or -0.07 < distance_ma20 <= -0.04:
            score += 15

    if pd.notna(drawdown60):
        if -0.25 <= drawdown60 <= -0.10:
            score += 20
        elif -0.35 <= drawdown60 < -0.25:
            score += 10

    return min(score, 100.0)


def reversal_score(latest: pd.Series) -> float:
    score = 0.0

    close = latest.get("Close", np.nan)
    ma5 = latest.get("ma5", np.nan)
    ma20 = latest.get("ma20", np.nan)
    volume_ratio = latest.get("volume_ratio", np.nan)
    rsi = latest.get("rsi14", np.nan)

    if pd.notna(close) and pd.notna(ma5) and close > ma5:
        score += 30
    if pd.notna(close) and pd.notna(ma20) and close > ma20:
        score += 10
    if bool(latest.get("reversal_confirmed", False)):
        score += 30
    if pd.notna(volume_ratio):
        if volume_ratio >= 1.5:
            score += 20
        elif volume_ratio >= 1.1:
            score += 10
    if pd.notna(rsi) and pd.notna(rsi):
        prev_rsi = latest.get("prev_rsi14", np.nan)
        if pd.notna(prev_rsi) and rsi > prev_rsi:
            score += 10

    return min(score, 100.0)


def build_signal(latest: pd.Series) -> dict:
    """Return a transparent signal score and gate flags for the latest bar."""
    over_score = oversold_score(latest)
    rev_score = reversal_score(latest)

    # Fundamental is deliberately a separate gate. Until revenue/financial data is wired in,
    # callers can pass fundamental_pass=False to block trades.
    fundamental_pass = bool(latest.get("fundamental_pass", True))
    reversal_pass = bool(latest.get("reversal_confirmed", False))

    score = round(over_score * 0.55 + rev_score * 0.45, 2)

    return {
        "score": score,
        "oversold_score": round(over_score, 2),
        "reversal_score": round(rev_score, 2),
        "fundamental_pass": fundamental_pass,
        "reversal_pass": reversal_pass,
        "buy_zone": score >= 70 and fundamental_pass and reversal_pass,
    }
