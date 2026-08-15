import ta

def build_features(df):
    df = df.copy()

    df["ma5"] = ta.trend.sma_indicator(df["Close"], 5)
    df["ma20"] = ta.trend.sma_indicator(df["Close"], 20)
    df["ma60"] = ta.trend.sma_indicator(df["Close"], 60)

    df["vol20"] = df["Volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["Volume"] / df["vol20"]
    )

    df["high60"] = (
        df["Close"]
        .rolling(60)
        .max()
    )

    df["breakout_ratio"] = (
        df["Close"] / df["high60"]
    )

    return df