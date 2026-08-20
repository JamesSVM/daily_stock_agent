import os
import time
import sqlite3
import requests
import pandas as pd

# ==========================================
# 1. 初始化法人資料庫表
# ==========================================
def init_institutional_db(db_path="data/database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 建立 institutional_data 資料表，使用複合主鍵避免重複寫入
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutional_data (
            stock_id TEXT,
            date TEXT,
            foreign_buy INTEGER,
            trust_buy INTEGER,
            PRIMARY KEY (stock_id, date)
        )
    """)
    conn.commit()
    conn.close()
    print("✅ 資料庫 institutional_data 資料表初始化完成！")

# ==========================================
# 2. 獲取與清洗 FinMind 法人資料
# ==========================================
def fetch_institutional_data(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": str(stock_id),
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        response = requests.get(url, params=parameter, timeout=10)
        data = response.json()

        if data.get("msg") == "success" and data.get("data"):
            df = pd.DataFrame(data["data"])

            # 計算每日每筆的淨買賣超股數
            df['net_buy'] = df['buy'] - df['sell']

            # 1. 篩選外資 (包含 Foreign_Investor 與 Foreign_Dealer_Self)
            foreign_mask = df['name'].str.contains('Foreign', case=False, na=False)
            df_foreign = df[foreign_mask].groupby('date')['net_buy'].sum().reset_index()
            df_foreign.rename(columns={'net_buy': 'foreign_buy'}, inplace=True)

            # 2. 篩選投信 (Investment_Trust)
            trust_mask = df['name'].str.contains('Investment_Trust', case=False, na=False)
            df_trust = df[trust_mask].groupby('date')['net_buy'].sum().reset_index()
            df_trust.rename(columns={'net_buy': 'trust_buy'}, inplace=True)

            # 3. 合併外資與投信數據
            merged = pd.merge(df_foreign, df_trust, on='date', how='outer').fillna(0)
            merged['stock_id'] = stock_id

            # 確保欄位型別與順序正確
            merged['foreign_buy'] = merged['foreign_buy'].astype(int)
            merged['trust_buy'] = merged['trust_buy'].astype(int)

            return merged[['stock_id', 'date', 'foreign_buy', 'trust_buy']]

    except Exception as e:
        print(f"❌ 抓取 {stock_id} 法人資料發生錯誤: {e}")

    return pd.DataFrame()

# ==========================================
# 3. 寫入 SQLite (INSERT OR REPLACE)
# ==========================================
def save_institutional_to_sqlite(df, db_path="data/database.db"):
    if df.empty:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    records = df.to_records(index=False).tolist()

    cursor.executemany("""
        INSERT OR REPLACE INTO institutional_data
        (stock_id, date, foreign_buy, trust_buy)
        VALUES (?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_institutional_db()

    # 台灣 50 (0050) + 中型 100 (0051) 的 150 檔成分股代號
    TARGET_STOCKS = [
        "2330", "2317", "2454", "2382", "2308", "2881", "2882", "2891", "2886", "2603",
        "3711", "3008", "1301", "1303", "2002", "1216", "2884", "2885", "2892", "2412",
        "2880", "2883", "2327", "2887", "5871", "1101", "2303", "2890", "2912", "5880",
        "2207", "2395", "2408", "3034", "3045", "4904", "6505", "1229", "1402", "2105",
        "2301", "2344", "2609", "2615", "3037", "3231", "3661", "6669", "8046", "9904",
        "1102", "1319", "1476", "1504", "1536", "1560", "1590", "1605", "1707", "1717",
        "1722", "1789", "1795", "1802", "1907", "2006", "2014", "2027", "2049", "2059",
        "2106", "2201", "2204", "2206", "2231", "2313", "2323", "2324", "2337", "2345",
        "2347", "2352", "2353", "2356", "2362", "2368", "2376", "2377", "2379", "2383",
        "2385", "2392", "2404", "2409", "2439", "2449", "2451", "2474", "2504", "2515",
        "2520", "2542", "2545", "2548", "2606", "2607", "2610", "2618", "2633", "2637",
        "2707", "2723", "2801", "2809", "2812", "2834", "2845", "2850", "2851", "2852",
        "2888", "2889", "2903", "2915", "3005", "3017", "3019", "3023", "3033", "3044",
        "3189", "3324", "3443", "3481", "3532", "3592", "3653", "3702", "3704", "3706",
        "4919", "4938", "4958", "5347", "5483", "5522", "6176", "6239", "6269", "6285"
    ]

    START_DATE = "2020-01-01"
    END_DATE = "2026-08-17"

    print(f"\n🚀 開始下載 150 檔法人資料...")
    for idx, stock in enumerate(TARGET_STOCKS, 1):
        print(f"[{idx}/150] 正在抓取 {stock} 法人籌碼...", end=" ")
        df_inst = fetch_institutional_data(stock, START_DATE, END_DATE)

        if not df_inst.empty:
            save_institutional_to_sqlite(df_inst)
            print(f"✅ 寫入 {len(df_inst)} 筆。")
        else:
            print("⚠️ 無資料或抓取失敗。")

        # 安全延遲，確保不會被 API 阻擋
        time.sleep(2.5)

    print("\n🎉 150 檔法人籌碼歷史資料庫建檔完成！")