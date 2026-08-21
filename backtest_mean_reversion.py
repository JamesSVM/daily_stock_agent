import sqlite3

import pandas as pd

from crawler.price import get_price
from engine.backtest_mean_reversion import simulate_stock
from engine.metrics import evaluate_backtest
from engine.market_regime import build_market_proxy


DB_PATH = "data/database.db"


def get_stock_ids(conn):
    query = """
        SELECT DISTINCT stock_id
        FROM daily_price
        ORDER BY stock_id
    """
    rows = conn.execute(query).fetchall()
    return [row[0] for row in rows]


def main():
    all_trades = []
    stock_data = {}

    conn = sqlite3.connect(DB_PATH)

    try:
        stock_ids = get_stock_ids(conn)

        print(f"Universe: {len(stock_ids)} stocks")

        # --------------------------------------------------
        # 1. Load all stock data
        # --------------------------------------------------

        for stock_id in stock_ids:
            df = get_price(stock_id, period="5y")

            if df is None or len(df) < 100:
                print(f"{stock_id}: insufficient data")
                continue

            stock_data[stock_id] = df

        print(f"\nLoaded: {len(stock_data)} stocks")

        # --------------------------------------------------
        # 2. Build market regime
        # --------------------------------------------------

        print("\nBuilding market proxy...")

        market = build_market_proxy(stock_data)

        if market.empty:
            print("Market proxy is empty.")
            return

        print("\nMarket Regime Distribution:")

        print(
            market["regime"]
            .value_counts()
            .to_dict()
        )

        # --------------------------------------------------
        # 3. Run stock backtest
        # --------------------------------------------------

        for stock_id, df in stock_data.items():

            trades = simulate_stock(
                df,
                stock_id=stock_id,
                hold_days=10,
                stop_loss=0.05,
                take_profit=0.10,
                min_score=70,
                market=market,
                allowed_regimes={"bull", "neutral"},
            )

            all_trades.extend(trades)

            print(f"{stock_id}: {len(trades)} trades")

        # --------------------------------------------------
        # 4. Evaluate
        # --------------------------------------------------

        summary = evaluate_backtest(all_trades)

        print("\n=== Mean Reversion V1.2 Backtest ===")

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

        # --------------------------------------------------
        # 4.1 Alpha Analysis
        # --------------------------------------------------

        trades_df = pd.DataFrame(all_trades)

        if not trades_df.empty:

            print("\n[Alpha Analysis]")

            print(
                f"Avg Market Return: "
                f"{trades_df['market_return'].mean() * 100:.2f}%"
            )

            print(
                f"Avg Alpha: "
                f"{trades_df['alpha'].mean() * 100:.2f}%"
            )

            print(
                f"Median Alpha: "
                f"{trades_df['alpha'].median() * 100:.2f}%"
            )

            print(
                f"Alpha > 0: "
                f"{(trades_df['alpha'] > 0).mean() * 100:.2f}%"
            )

            print(
                f"Alpha > 2%: "
                f"{(trades_df['alpha'] > 0.02).mean() * 100:.2f}%"
            )

            print(
                f"Alpha > 5%: "
                f"{(trades_df['alpha'] > 0.05).mean() * 100:.2f}%"
            )

            print("\n[Market Regime Alpha]")

            regime_analysis = (
                trades_df
                .groupby("market_regime")
                .agg(
                    trades=("alpha", "count"),
                    avg_return=("return", "mean"),
                    avg_market_return=(
                        "market_return",
                        "mean",
                    ),
                    avg_alpha=("alpha", "mean"),
                    win_rate=(
                        "return",
                        lambda x: (x > 0).mean(),
                    ),
                )
            )

            regime_analysis["avg_return"] *= 100
            regime_analysis["avg_market_return"] *= 100
            regime_analysis["avg_alpha"] *= 100
            regime_analysis["win_rate"] *= 100

            print(
                regime_analysis.round(2)
            )

            print("\n[Relative Strength Analysis]")

            rs = trades_df[
                "relative_strength_20d"
            ]

            bins = [
                float("-inf"),
                -0.10,
                -0.05,
                0.00,
                0.05,
                0.10,
                float("inf"),
            ]

            labels = [
                "<-10%",
                "-10~-5%",
                "-5~0%",
                "0~5%",
                "5~10%",
                ">10%",
            ]

            trades_df["rs_bucket"] = pd.cut(
                rs,
                bins=bins,
                labels=labels,
            )

            rs_analysis = (
                trades_df
                .groupby(
                    "rs_bucket",
                    observed=False,
                )
                .agg(
                    trades=("alpha", "count"),
                    avg_return=("return", "mean"),
                    avg_alpha=("alpha", "mean"),
                    win_rate=(
                        "return",
                        lambda x: (x > 0).mean(),
                    ),
                )
            )

            rs_analysis["avg_return"] *= 100
            rs_analysis["avg_alpha"] *= 100
            rs_analysis["win_rate"] *= 100

            print(
                rs_analysis.round(2)
            )


        # --------------------------------------------------
        # 5. Save trades
        # --------------------------------------------------

        if all_trades:

            pd.DataFrame(all_trades).to_csv(
                "trades_mean_reversion_v1_3.csv",
                index=False
            )

            print(
                "\nSaved: trades_mean_reversion_v1_3.csv"
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()