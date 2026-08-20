import requests
import pandas as pd

def fetch_taiwan_stock_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """透過 FinMind 獲取台股歷史價量資料"""
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": str(stock_id),
        "start_date": start_date,
        "end_date": end_date,
    }

    response = requests.get(url, params=parameter)
    data = response.json()

    if data["msg"] == "success" and data["data"]:
        df = pd.DataFrame(data["data"])

        # 欄位重新命名以對應我們的 SQLite schema
        df = df.rename(columns={
            "stock_id": "stock_id",
            "date": "date",
            "open": "open",
            "max": "high",
            "min": "low",
            "close": "close",
            "Trading_Volume": "volume"
        })

        # 確保順序與資料庫一致
        return df[["stock_id", "date", "open", "high", "low", "close", "volume"]]

    return pd.DataFrame()