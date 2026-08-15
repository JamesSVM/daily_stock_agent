import yfinance as yf
import pandas as pd

def get_price(stock_id, period="6mo"):
    ticker = f"{stock_id}.TW"
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    df = df.reset_index()
    return df

if __name__ == "__main__":
    df = get_price("2330")
    print(df.tail())