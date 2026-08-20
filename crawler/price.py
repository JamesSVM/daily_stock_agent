import pandas as pd
import yfinance as yf


def get_price(stock_id, period="6mo"):
    ticker = f"{stock_id}.TW"
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    return df


if __name__ == "__main__":
    df = get_price("2330")
    print(df.tail())
    print(df.columns.tolist())
