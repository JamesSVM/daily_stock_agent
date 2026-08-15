import yfinance as yf
import pandas as pd

def _fetch_yf_data(ticker, period="6mo"):
    """內部輔助函式：負責實際下載與整理 DataFrame"""
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    # 修正 yfinance 新版的 MultiIndex 欄位結構
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 移除重複欄位（若有）並將 Date 轉為一般欄位
    df = df.reset_index()
    return df

def get_price(stock_id, period="6mo"):
    """
    支援台股上市 (.TW)、上櫃 (.TWO) 自動判斷與美股代碼查詢
    
    :param stock_id: 股票代碼 (str 或 int)，例如 '2330', '6488', 'AAPL', '0050'
    :param period: 查詢區間，預設 '6mo' (可填 '1mo', '1y', '5y', 'max' 等)
    :return: pd.DataFrame 或 None
    """
    symbol = str(stock_id).strip().upper()

    # 情況 1：輸入純數字代碼（預設為台股，依序嘗試上市與上櫃）
    if symbol.isdigit():
        # 優先嘗試上市 (.TW)
        df = _fetch_yf_data(f"{symbol}.TW", period=period)
        if df is not None:
            return df
        
        # 上市找不到，嘗試上櫃 (.TWO)
        df = _fetch_yf_data(f"{symbol}.TWO", period=period)
        if df is not None:
            return df
        
        return None

    # 情況 2：美股代碼或已包含市場後綴 (如 'AAPL', 'TSLA', '2330.TW')
    return _fetch_yf_data(symbol, period=period)