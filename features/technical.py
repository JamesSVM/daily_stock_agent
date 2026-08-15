import ta

def add_features(df):
    df = df.copy()

    df["ma5"] = ta.trend.sma_indicator(df["Close"], 5)
    df["ma20"] = ta.trend.sma_indicator(df["Close"], 20)
    df["ma60"] = ta.trend.sma_indicator(df["Close"], 60)

    df["vol20"] = df["Volume"].rolling(20).mean()

    return df

def score_technical(df):
    latest = df.iloc[-1]

    score = 0

    if latest["Close"] > latest["ma5"]:
        score += 20

    if latest["ma5"] > latest["ma20"]:
        score += 20

    if latest["ma20"] > latest["ma60"]:
        score += 20

    if latest["Volume"] > latest["vol20"] * 1.5:
        score += 20

    high60 = df["Close"].tail(60).max()

    if latest["Close"] >= high60 * 0.98:
        score += 20

    return score