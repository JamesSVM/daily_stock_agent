from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from engine.backtest_mean_reversion import simulate_stock
from engine.portfolio_backtest import CostModel, simulate_portfolio

DB_PATH = "data/database.db"
REPORT_DIR = "reports"

INITIAL_CAPITAL = 1_000_000.0
ALLOCATION_PCT = 0.10
MAX_POSITIONS = 10
HOLD_DAYS = 10
STOP_LOSS = 0.05
TAKE_PROFIT = 0.10
MIN_SCORE = 70.0


def load_stock_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT stock_id FROM daily_price ORDER BY stock_id"
    ).fetchall()
    return [str(row[0]) for row in rows]


def load_stock_data(conn: sqlite3.Connection, stock_id: str) -> pd.DataFrame:
    query = """
        SELECT
            date AS Date,
            open AS Open,
            high AS High,
            low AS Low,
            close AS Close,
            volume AS Volume
        FROM daily_price
        WHERE stock_id = ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=[stock_id])
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    return df.sort_values("Date").set_index("Date")


def load_trade_prices(
    conn: sqlite3.Connection,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["stock_id", "date", "close"])

    stock_ids = sorted(trades["stock_id"].astype(str).unique().tolist())
    start_date = pd.to_datetime(trades["entry_date"]).min()
    end_date = pd.to_datetime(trades["exit_date"]).max()
    placeholders = ",".join("?" for _ in stock_ids)

    query = f"""
        SELECT stock_id, date, close
        FROM daily_price
        WHERE stock_id IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY date, stock_id
    """

    return pd.read_sql_query(
        query,
        conn,
        params=[*stock_ids, str(start_date.date()), str(end_date.date())],
    )


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        stock_ids = load_stock_ids(conn)
        print(f"Universe: {len(stock_ids)} stocks")

        all_trades: list[dict] = []

        for index, stock_id in enumerate(stock_ids, start=1):
            df = load_stock_data(conn, stock_id)
            if len(df) < 100:
                print(f"[{index:03d}/{len(stock_ids)}] {stock_id}: insufficient data")
                continue

            trades = simulate_stock(
                df,
                stock_id=stock_id,
                hold_days=HOLD_DAYS,
                stop_loss=STOP_LOSS,
                take_profit=TAKE_PROFIT,
                min_score=MIN_SCORE,
            )
            all_trades.extend(trades)
            print(f"[{index:03d}/{len(stock_ids)}] {stock_id}: {len(trades)} trades")

        trades_df = pd.DataFrame(all_trades)
        if trades_df.empty:
            print("No trades generated.")
            return

        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"], errors="coerce")
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"], errors="coerce")
        prices_df = load_trade_prices(conn, trades_df)

    costs = CostModel(
        buy_commission=0.001425,
        sell_commission=0.001425,
        sell_tax=0.003,
        buy_slippage=0.0005,
        sell_slippage=0.0005,
    )

    equity_curve, fills_df, rejected_df, metrics = simulate_portfolio(
        trades_df,
        prices_df,
        initial_capital=INITIAL_CAPITAL,
        allocation_pct=ALLOCATION_PCT,
        max_positions=MAX_POSITIONS,
        costs=costs,
    )

    output_dir = Path(REPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(output_dir / "trades_mean_reversion_v12.csv", index=False)
    prices_df.to_csv(output_dir / "portfolio_prices_v12.csv", index=False)
    equity_curve.to_csv(output_dir / "portfolio_equity_v12.csv", index=False)
    fills_df.to_csv(output_dir / "portfolio_fills_v12.csv", index=False)
    rejected_df.to_csv(output_dir / "portfolio_rejections_v12.csv", index=False)
    pd.DataFrame([metrics]).to_csv(
        output_dir / "portfolio_summary_v12.csv",
        index=False,
    )

    print("\n=== V1.2 Portfolio Backtest ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nReports saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
