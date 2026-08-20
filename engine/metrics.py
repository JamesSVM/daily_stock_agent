import pandas as pd

def evaluate(trades):
    if len(trades) == 0:
        return {}
        
    win_rate = (trades["return"] > 0).mean()
    avg_return = trades["return"].mean()
    
    # 避免全是虧損或全是獲利時的計算錯誤
    win_returns = trades[trades["return"] > 0]["return"]
    loss_returns = trades[trades["return"] <= 0]["return"]
    
    avg_win = win_returns.mean() if not win_returns.empty else 0
    avg_loss = loss_returns.mean() if not loss_returns.empty else 0
    
    expectancy = (avg_win * win_rate) + (avg_loss * (1 - win_rate))
    
    result = {
        "trades": len(trades),
        "win_rate": round(float(win_rate * 100), 2),
        "avg_return": round(float(avg_return * 100), 2),
        "expectancy": round(float(expectancy * 100), 2)
    }
    
    # 計算各別出場原因的統計數據
    if "exit_reason" in trades.columns:
        exit_stats = {}
        grouped = trades.groupby("exit_reason")
        
        for name, group in grouped:
            count = len(group)
            pct = count / len(trades)
            group_avg_ret = group["return"].mean()
            group_win_rate = (group["return"] > 0).mean()
            
            exit_stats[name] = {
                "次數": count,
                "佔比": f"{round(pct * 100, 2)}%",
                "該類別勝率": f"{round(group_win_rate * 100, 2)}%",
                "該類別平均報酬": f"{round(group_avg_ret * 100, 2)}%"
            }
            
        result["exit_stats"] = exit_stats
        
    return result