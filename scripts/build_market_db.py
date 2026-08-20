import os
import sqlite3
import time
import requests
import pandas as pd

# ==========================================
# 1. 初始化資料庫 (確保 Schema 正確)
# ==========================================
def init_db(db_path="data/database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 複合主鍵 (stock_id, date) 確保資料不重複
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_price (
            stock_id TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (stock_id, date)
        )
    """)
    conn.commit()
    conn.close()
    print("✅ 資料庫 daily_price 資料表初始化完成！")

# ==========================================
# 2. FinMind API 爬蟲
# ==========================================
def fetch_finmind_price(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": str(stock_id),
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        response = requests.get(url, params=parameter, timeout=10)
        data = response.json()

        if data.get("msg") == "success" and data.get("data"):
            df = pd.DataFrame(data["data"])

            # 對齊 SQLite Schema
            df = df.rename(columns={
                "max": "high",
                "min": "low",
                "Trading_Volume": "volume"
            })
            return df[["stock_id", "date", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"❌ 抓取 {stock_id} 發生錯誤: {e}")

    return pd.DataFrame()

# ==========================================
# 3. 寫入 SQLite (INSERT OR REPLACE)
# ==========================================
def save_to_sqlite(df, db_path="data/database.db"):
    if df.empty:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    records = df.to_records(index=False).tolist()

    # 使用 INSERT OR REPLACE 進行 Upsert
    cursor.executemany("""
        INSERT OR REPLACE INTO daily_price
        (stock_id, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()

# ==========================================
# 主程式：批次執行 150 檔
# ==========================================
if __name__ == "__main__":
    init_db()

    # 台灣 50 (0050) + 中型 100 (0051) 的 150 檔代號清單

    TARGET_STOCKS = [
    # 台灣 50 (0050) 成分股 (大型股代表)
    "2330", "2317", "2454", "2382", "2308", "2881", "2882", "2891", "2886", "2603",
    "3711", "3008", "1301", "1303", "2002", "1216", "2884", "2885", "2892", "2412",
    "2880", "2883", "2327", "2887", "5871", "1101", "2303", "2890", "2912", "5880",
    "2207", "2395", "2408", "3034", "3045", "4904", "6505", "1229", "1402", "2105",
    "2301", "2344", "2609", "2615", "3037", "3231", "3661", "6669", "8046", "9904",

    # 中型 100 (0051) 成分股 (中型股代表)
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

    # 設定你要回測的歷史長度 (抓取過去 5 年以上的資料能讓 AI 的勝率統計更有參考價值)
    START_DATE = "2020-01-01"
    END_DATE = "2026-08-17"

    print(f"\n🚀 開始批次下載，共 {len(TARGET_STOCKS)} 檔股票...")

    for stock_id in TARGET_STOCKS:
        print(f"正在抓取 {stock_id} ...", end=" ")

        df = fetch_finmind_price(stock_id, START_DATE, END_DATE)

        if not df.empty:
            save_to_sqlite(df)
            print(f"✅ 寫入 {len(df)} 筆。")
        else:
            print("⚠️ 無資料或抓取失敗。")

        # 關鍵：FinMind 免費版對請求頻率有限制，強烈建議每次抓取後暫停 2-3 秒
        time.sleep(2.5)

    print("\n🎉 所有股票歷史資料庫建檔完成！")