import sqlite3
import os

def init_db(db_path="data/database.db"):
    # 確保資料夾存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 建立 daily_price 資料表
    # 使用 stock_id 和 date 作為複合主鍵，防止重複抓取導致資料重複
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
    print("資料庫初始化完成！")

if __name__ == "__main__":
    init_db()