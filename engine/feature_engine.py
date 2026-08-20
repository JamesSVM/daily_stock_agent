import ta

def build_features(df):
    df = df.copy()
    
    # 均線系統 (補上投信最愛的 10 日線)
    df["ma5"] = ta.trend.sma_indicator(df["Close"], 5)
    df["ma10"] = ta.trend.sma_indicator(df["Close"], 10)
    df["ma20"] = ta.trend.sma_indicator(df["Close"], 20)
    df["ma60"] = ta.trend.sma_indicator(df["Close"], 60)
    
    # 爆發力量能
    df["vol20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["vol20"]
    
    # 突破特徵
    df["high60"] = df["Close"].rolling(60).max()
    df["breakout_ratio"] = df["Close"] / df["high60"]
    
    # 風險特徵：乖離率 (股價距離均線的百分比)
    df["bias_ma10"] = (df["Close"] - df["ma10"]) / df["ma10"]
    df["bias_ma20"] = (df["Close"] - df["ma20"]) / df["ma20"]
    
    # 2. 籌碼特徵 (Institutional Factors)
    # 假設 df 中已經 JOIN 了外資(Foreign_Buy)與投信(Trust_Buy)的每日淨買賣超張數
    
    # 計算近 3 日投信與外資的累積買賣超
    df["Trust_Buy_3D"] = df["Trust_Buy"].rolling(window=3).sum()
    df["Foreign_Buy_3D"] = df["Foreign_Buy"].rolling(window=3).sum()
    
    # 計算近 5 日投信「連續買超」的天數 (布林值轉整數後 rolling sum)
    df["Trust_Consecutive_Buy"] = (df["Trust_Buy"] > 0).astype(int).rolling(window=5).sum()


    return df