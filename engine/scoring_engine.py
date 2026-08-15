def calculate_score(row):

    score = 0

    if row["Close"] > row["ma5"]:
        score += 20

    if row["ma5"] > row["ma20"]:
        score += 20

    if row["ma20"] > row["ma60"]:
        score += 20

    if row["volume_ratio"] > 1.5:
        score += 20

    if row["breakout_ratio"] > 0.98:
        score += 20

    return score