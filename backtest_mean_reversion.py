from crawler.price import get_price
from engine.backtest_mean_reversion import simulate_stock
from engine.metrics import evaluate_backtest

WATCHLIST = [
    "2330", "2317", "2382", "2454", "3017", "3661", "2303", "2603"
]


def main():
    all_trades = []

    for stock_id in WATCHLIST:
        df = get_price(stock_id, period="2y")
        if df is None or len(df) < 100:
            print(f"{stock_id}: insufficient data")
            continue

        trades = simulate_stock(
            df,
            stock_id=stock_id,
            hold_days=10,
            stop_loss=0.05,
            take_profit=0.10,
            min_score=70,
        )
        all_trades.extend(trades)
        print(f"{stock_id}: {len(trades)} trades")

    
    summary = evaluate_backtest(all_trades)

    print("\n=== Mean Reversion Backtest ===")

    print("\n[Performance]")
    for key, value in summary["performance"].items():
        print(f"{key}: {value}")

    print("\n[Score Analysis]")
    for score_bucket, stats in summary["score_analysis"].items():
        print(f"{score_bucket}: {stats}")

    print("\n[Exit Analysis]")
    for exit_reason, stats in summary["exit_analysis"].items():
        print(f"{exit_reason}: {stats}")

    print("\n[Stock Analysis]")
    for stock_id, stats in summary["stock_analysis"].items():
        print(f"{stock_id}: {stats}")

    if all_trades:
        import pandas as pd
        pd.DataFrame(all_trades).to_csv("trades_mean_reversion.csv", index=False)
        print("\nSaved: trades_mean_reversion.csv")


if __name__ == "__main__":
    main()
