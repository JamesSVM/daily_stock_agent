def evaluate(trades):

    if len(trades) == 0:
        return {}

    win_rate = (trades["return"] > 0).mean()

    avg_return = trades["return"].mean()

    expectancy = (
        trades[trades["return"] > 0]["return"].mean() *
        win_rate +
        trades[trades["return"] <= 0]["return"].mean() *
        (1-win_rate)
    )

    return {
        "trades": len(trades),
        "win_rate": round(float(win_rate*100), 2),
        "avg_return": round(float(avg_return*100), 2),
        "expectancy": round(float(expectancy*100), 2)
    }