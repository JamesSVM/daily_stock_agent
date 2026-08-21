from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from engine.market_backtest import load_stock_data, load_stock_ids, run_market_backtest
from engine.portfolio_backtest import CostModel, simulate_portfolio

DB_PATH = "data/database.db"
REPORT_DIR = "reports"

INITIAL_CAPITAL = 1_000_000.0
ALLOCATION_PCT = 0.10
MAX_POSITIONS = 10


def load_prices_for_trades(
    trades: pd.DataFrame,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load close prices only for stocks and dates needed by the portfolio run."""
    if trades.empty:
        return pd.DataFrame(columns=["stock_id", "date", "close"])

    stock_ids = sorted(trades["stock_id"].astype(str).unique().tolist())
    start_date = pd.to_datetime(trades["entry_date"]).min()
    end_date = pd.to_datetime(trades["exit_date"]).max()

    placeholders = ",".join("?" for _ in stock_ids)
    query = f"""
        SELECT
            stock_id,
            date,
            close
        FROM daily_price
        WHERE stock_id IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY date, stock_id
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            query,
            conn,
            params=[*stock_ids, str(start_date.date()), str(end_date.date())],
        )


def main() -> None:
    stock_ids = load_stock_ids(DB_PATH)
    print(f"Universe: {len(stock_ids)} stocks")

    trades_df, diagnostics_df = run_market_backtest(
        stock_ids,
        db_path=DB_PATH,
        hold_days=10,
        stop_loss=0.05,
        take_profit=0.10,
        min_score=70.0,
        min_rows=100,
    )

    if trades_df.empty:
        print("No trades generated; portfolio backtest cannot run.")
        return

    prices_df = load_prices_for_trades(trades_df, DB_PATH)

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

    out = Path(REPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(out / "trades_mean_reversion_v12.csv", index=False)
    diagnostics_df.to_csv(out / "signal_diagnostics_v12.csv", index=False)
    prices_df.to_csv(out / "portfolio_prices_v12.csv", index=False)
    equity_curve.to_csv(out / "portfolio_equity_v12.csv", index=False)
    fills_df.to_csv(out / "portfolio_fills_v12.csv", index=False)
    rejected_df.to_csv(out / "portfolio_rejections_v12.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out / "portfolio_summary_v12.csv", index=False)

    print("\n=== V1.2 Portfolio Backtest ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nReports saved to: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
