import sqlite3
import time
from crawler.finmind_client import fetch_taiwan_stock_price

def save_to_sqlite(df, db_path="data/database.db"):
    if df.empty:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 將 DataFrame 轉為 tuple 列表
    records = df.to_records(index=False).tolist()

    # 使用 INSERT OR REPLACE 避免重複寫入報錯，直接覆蓋舊資料
    cursor.executemany("""
        INSERT OR REPLACE INTO daily_price
        (stock_id, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()

# 測試單檔寫入
if __name__ == "__main__":
    test_stock = "2382"
    print(f"正在下載 {test_stock}...")
    df = fetch_taiwan_stock_price(test_stock, "2022-01-01", "2026-08-15")
    save_to_sqlite(df)
    print(f"{test_stock} 下載並寫入完成，共 {len(df)} 筆資料。")