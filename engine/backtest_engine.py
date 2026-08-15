import pandas as pd

def run_backtest(
    df,
    score_func,
    holding_days=5,
    threshold=80
):

    trades = []

    for i in range(60, len(df)-holding_days):

        row = df.iloc[i]

        score = score_func(row)

        if score < threshold:
            continue

        entry = row["Close"]

        exit_price = df.iloc[
            i+holding_days
        ]["Close"]

        ret = (
            exit_price - entry
        ) / entry

        trades.append({
            "date": row["Date"],
            "score": score,
            "return": ret
        })

    return pd.DataFrame(trades)