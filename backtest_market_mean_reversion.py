from __future__ import annotations

from engine.market_backtest import (
    build_analysis_tables,
    load_stock_ids,
    run_market_backtest,
    save_reports,
    summarize_trades,
)


DB_PATH = "data/database.db"
OUTPUT_DIR = "reports"

HOLD_DAYS = 10
STOP_LOSS = 0.05
TAKE_PROFIT = 0.10
MIN_SCORE = 70.0
MIN_ROWS = 100


def main() -> None:
    stock_ids = load_stock_ids(DB_PATH)

    print("=" * 70)
    print("Mean Reversion Backtest V1.1 - 0050 + 0051 universe")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print(f"Stocks found: {len(stock_ids)}")
    print(f"Hold days: {HOLD_DAYS}")
    print(f"Stop loss: {STOP_LOSS:.0%}")
    print(f"Take profit: {TAKE_PROFIT:.0%}")
    print(f"Minimum score: {MIN_SCORE}")

    if not stock_ids:
        raise RuntimeError(
            "No stocks found in daily_price. "
            "Please check data/database.db."
        )

    trades_df, diagnostics_df = run_market_backtest(
        stock_ids=stock_ids,
        db_path=DB_PATH,
        hold_days=HOLD_DAYS,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        min_score=MIN_SCORE,
        min_rows=MIN_ROWS,
    )

    summary = summarize_trades(trades_df)
    analysis_tables = build_analysis_tables(trades_df)

    print("\n=== Overall Performance ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Signal Diagnostics ===")
    if diagnostics_df.empty:
        print("No diagnostic rows.")
    else:
        print(
            f"stocks: {len(diagnostics_df)} | "
            f"stocks_with_trades: "
            f"{int((diagnostics_df['trades'] > 0).sum())} | "
            f"total_buy_zone_days: "
            f"{int(diagnostics_df['buy_zone_days'].sum())}"
        )
        print("\nTop stocks by buy-zone frequency:")
        print(
            diagnostics_df.sort_values(
                "buy_zone_rate_pct", ascending=False
            )[
                [
                    "stock_id",
                    "eligible_days",
                    "buy_zone_days",
                    "buy_zone_rate_pct",
                    "trades",
                ]
            ].head(15).to_string(index=False)
        )

    print("\n=== Score Analysis ===")
    score_table = analysis_tables["score_analysis"]
    if score_table.empty:
        print("No score analysis.")
    else:
        print(score_table.to_string(index=False))

    print("\n=== Exit Analysis ===")
    exit_table = analysis_tables["exit_analysis"]
    if exit_table.empty:
        print("No exit analysis.")
    else:
        print(exit_table.to_string(index=False))

    print("\n=== Stock Analysis (top / bottom) ===")
    stock_table = analysis_tables["stock_analysis"]
    if stock_table.empty:
        print("No stock analysis.")
    else:
        print("\nTop 10:")
        print(stock_table.head(10).to_string(index=False))
        print("\nBottom 10:")
        print(stock_table.tail(10).sort_values("avg_return_pct").to_string(index=False))

    save_reports(
        output_dir=OUTPUT_DIR,
        trades_df=trades_df,
        diagnostics_df=diagnostics_df,
        summary=summary,
        analysis_tables=analysis_tables,
    )

    print(f"\nReports saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
