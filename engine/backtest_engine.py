import pandas as pd

def run_backtest(df, score_func, threshold=80):
    # 重設 index 確保後續利用 index 計算持有天數時連續不中斷
    df = df.copy().reset_index(drop=True)
    df["score"] = score_func(df)

    # ==========================================
    # 預先計算兩種「絕對出場」的布林值條件 (極速向量化)
    # ==========================================
    # 條件 1: 跌破 10 日均線
    cond_ma10 = df["Close"] < df["ma20"]

    # 條件 2: 高檔爆大量收黑K
    # 定義：收盤小於開盤(黑K) + 成交量大於2倍月均量(爆量) + 股價在月線上(高檔)
    cond_black_k = (df["Close"] < df["Open"]) & (df["Volume"] > df["vol20"] * 2) & (df["Close"] > df["ma20"])

    signals = df[df["score"] >= threshold]
    trades = []

    # 針對少數觸發進場訊號的日子，進行未來的動態追蹤
    for idx, entry_row in signals.iterrows():
        entry_idx = idx
        entry_date = entry_row["Date"]
        entry_price = entry_row["Close"]

        # 截取進場日之後的「未來資料」來判定出場
        future_df = df.iloc[entry_idx + 1:]

        if future_df.empty:
            continue

        # ==========================================
        # 計算條件 3: 動態移動停損 (最高點回檔 7%)
        # ==========================================
        # 利用 cummax() 瞬間算出進場後每一天的「歷史最高價」
        running_max = future_df["High"].cummax().clip(lower=entry_price)
        cond_trailing_stop = future_df["Close"] < (running_max * 0.93)

        # 合併三道防線 (只要觸發任何一個即為 True)
        exit_mask = cond_ma10.loc[future_df.index] | cond_black_k.loc[future_df.index] | cond_trailing_stop

        # 找出觸發出場條件的日子
        exit_dates = future_df[exit_mask]

        if not exit_dates.empty:
            # 取第一天觸發的日期為出場日
            exit_idx = exit_dates.index[0]
            exit_row = future_df.loc[exit_idx]

            # 標記是哪一道防線觸發出場 (對事後覆盤極度重要)
            if cond_black_k.loc[exit_idx]:
                exit_reason = "高檔爆量黑K"
            elif cond_trailing_stop.loc[exit_idx]:
                exit_reason = "最高點回檔7%"
            else:
                exit_reason = "跌破MA10"
        else:
            # 若運氣極佳，到資料庫最後一天都沒破線，則以最後一天結算
            exit_row = future_df.iloc[-1]
            exit_reason = "資料終止"

        exit_price = exit_row["Close"]
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "date": entry_date,             # 進場日
            "exit_date": exit_row["Date"],  # 出場日
            "score": entry_row["score"],
            "return": ret,
            "hold_days": exit_row.name - entry_idx, # 持有交易日天數
            "exit_reason": exit_reason      # 出場原因
        })

    return pd.DataFrame(trades)