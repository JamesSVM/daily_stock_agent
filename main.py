import sqlite3
import pandas as pd
from engine.feature_engine import build_features
from engine.scoring_engine import calculate_score
from engine.backtest_engine import run_backtest
from engine.metrics import evaluate

def load_all_market_data(db_path="data/database.db"):
    """從 SQLite 透過 LEFT JOIN 載入全市場價量與法人籌碼資料"""
    conn = sqlite3.connect(db_path)

    # 使用 LEFT JOIN 合併每日價量與法人買賣超，並將無法人進出的日子補 0
    query = """
        SELECT
            p.stock_id,
            p.date as Date,
            p.open as Open,
            p.high as High,
            p.low as Low,
            p.close as Close,
            p.volume as Volume,
            COALESCE(i.foreign_buy, 0) as Foreign_Buy,
            COALESCE(i.trust_buy, 0) as Trust_Buy
        FROM daily_price p
        LEFT JOIN institutional_data i
            ON p.stock_id = i.stock_id AND p.date = i.date
        ORDER BY p.stock_id, p.date ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    print("📥 正在從 SQLite 載入全市場歷史價量與籌碼資料...")
    market_df = load_all_market_data()

    if market_df.empty:
        print("❌ 找不到資料，請確認資料庫建檔是否成功。")
        exit()

    print(f"✅ 成功載入 {len(market_df)} 筆全市場日K與籌碼資料。")
    print("⚙️ 正在進行批次特徵運算與回測 (已套用法人籌碼濾網)...")

    all_trades = []

    for stock_id, stock_df in market_df.groupby("stock_id"):
        # 1. 建立特徵 (包含價量指標與籌碼的滾動計算)
        featured_df = build_features(stock_df)

        # 2. 執行回測 (使用複合風控一票否決，門檻 80 分)
        # 注意：這裡採用動態出場機制，已無 holding_days 參數
        trades = run_backtest(featured_df, calculate_score, threshold=80)

        if not trades.empty:
            trades["stock_id"] = stock_id
            all_trades.append(trades)

    if all_trades:
        final_trades_df = pd.concat(all_trades, ignore_index=True)

        print("\n" + "="*60)
        print(" 📊 全市場橫截面盲測結果 (籌碼濾網 + 動態出場)")
        print("="*60)

        metrics_result = evaluate(final_trades_df)
        print("[全市場整體績效評估]:")
        print(f"總交易次數: {metrics_result.get('trades', 0)}")
        print(f"總勝率: {metrics_result.get('win_rate', 0)}%")
        print(f"平均報酬: {metrics_result.get('avg_return', 0)}%")
        print(f"期望值: {metrics_result.get('expectancy', 0)}%\n")

        if "exit_stats" in metrics_result:
            print("[各防線出場統計]:")
            for reason, stats in metrics_result["exit_stats"].items():
                print(f"🔹 {reason}:")
                print(f"   觸發次數: {stats['次數']} ({stats['佔比']})")
                print(f"   勝率: {stats['該類別勝率']} | 平均報酬: {stats['該類別平均報酬']}\n")

        # 顯示獲利最高的前 5 筆交易，並印出出場原因
        print("[獲利最高的前 5 筆交易]:")
        top_5 = final_trades_df.sort_values(by="return", ascending=False).head(5)
        top_5["return"] = top_5["return"].apply(lambda x: f"{x*100:.2f}%")
        print(top_5[["stock_id", "date", "exit_date", "score", "return", "exit_reason"]].to_string(index=False))

    else:
        print("⚠️ 這段期間內全市場沒有觸發任何符合籌碼與技術面雙重標準的進場訊號。")