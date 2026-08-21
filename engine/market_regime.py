from __future__ import annotations

import pandas as pd


def _prepare_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stock data so all calculations are aligned by Date."""

    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce",
        )
        data = data.dropna(subset=["Date"])
        data = data.set_index("Date")

    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(
            data.index,
            errors="coerce",
        )
        data = data[~data.index.isna()]

    data = data.sort_index()

    return data


def build_market_proxy(
    stock_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build an equal-weighted market proxy using calendar dates.

    Each stock contributes equally to the daily market return.

    Important:
    - All stocks are aligned by actual trading date.
    - Missing stock data on a date is ignored.
    """

    daily_returns = []

    for stock_id, df in stock_data.items():

        data = _prepare_stock_data(df)

        if data.empty:
            continue

        if "Close" not in data.columns:
            continue

        close = pd.to_numeric(
            data["Close"],
            errors="coerce",
        )

        returns = close.pct_change()
        returns.name = stock_id

        daily_returns.append(returns)

    if not daily_returns:
        return pd.DataFrame()

    returns_df = pd.concat(
        daily_returns,
        axis=1,
        join="outer",
    )

    market_return = returns_df.mean(
        axis=1,
        skipna=True,
    )

    market = pd.DataFrame(
        index=market_return.index,
    )

    market["return"] = market_return
    market["close"] = (
        1 + market["return"].fillna(0)
    ).cumprod()

    market["ma20"] = (
        market["close"]
        .rolling(20)
        .mean()
    )

    market["ma60"] = (
        market["close"]
        .rolling(60)
        .mean()
    )

    market["return_20d"] = (
        market["close"]
        .pct_change(20)
    )

    market["regime"] = "neutral"

    bull = (
        (market["close"] > market["ma60"])
        & (market["ma20"] > market["ma60"])
    )

    bear = (
        (market["close"] < market["ma60"])
        & (market["ma20"] < market["ma60"])
    )

    market.loc[bull, "regime"] = "bull"
    market.loc[bear, "regime"] = "bear"

    return market


def get_market_regime(
    market: pd.DataFrame,
    date,
) -> str:
    """Return the market regime available on a given date."""

    if market.empty:
        return "neutral"

    date = pd.to_datetime(
        date,
        errors="coerce",
    )

    if pd.isna(date):
        return "neutral"

    if date not in market.index:
        return "neutral"

    regime = market.loc[date, "regime"]

    if pd.isna(regime):
        return "neutral"

    return str(regime)


def get_market_return(
    market: pd.DataFrame,
    start_date,
    end_date,
) -> float:
    """Calculate market return between two dates."""

    if market.empty:
        return 0.0

    start_date = pd.to_datetime(
        start_date,
        errors="coerce",
    )

    end_date = pd.to_datetime(
        end_date,
        errors="coerce",
    )

    if pd.isna(start_date) or pd.isna(end_date):
        return 0.0

    market_close = market["close"]

    start_values = market_close.loc[
        market_close.index >= start_date
    ]

    end_values = market_close.loc[
        market_close.index <= end_date
    ]

    if start_values.empty or end_values.empty:
        return 0.0

    start_price = float(start_values.iloc[0])
    end_price = float(end_values.iloc[-1])

    if start_price == 0:
        return 0.0

    return end_price / start_price - 1.0