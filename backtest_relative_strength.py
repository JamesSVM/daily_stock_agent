import sqlite3

import pandas as pd

from crawler.price import get_price
from engine.backtest_relative_strength import (
    simulate_stock_relative_strength,
)
from engine.market_regime import build_market_proxy


DB_PATH = "data/database.db"


def get_stock_ids(conn):
    rows = conn.execute(
        """
        SELECT DISTINCT stock_id
        FROM daily_price
        ORDER BY stock_id
        """
    ).fetchall()

    return [str(row[0]) for row in rows]


def main():
    conn = sqlite3.connect(DB_PATH)

    try:
        stock_ids = get_stock_ids(conn)

        print(
            f"Universe: {len(stock_ids)} stocks"
        )

        stock_data = {}

        for stock_id in stock_ids:

            df = get_price(
                stock_id,
                period="5y",
            )

            if df is None or len(df) < 100:
                continue

            stock_data[stock_id] = df

        print(
            f"Loaded: {len(stock_data)} stocks"
        )

        market = build_market_proxy(
            stock_data
        )

        if market.empty:
            print("Market proxy is empty.")
            return

        all_trades = []

        for stock_id, df in stock_data.items():

            trades = simulate_stock_relative_strength(
                df,
                stock_id=stock_id,
                market=market,
                hold_days=10,
                stop_loss=0.05,
                take_profit=0.10,
                min_score=70.0,
                allowed_regimes={
                    "bull",
                    "neutral",
                },
            )

            all_trades.extend(trades)

            if trades:
                print(
                    f"{stock_id}: "
                    f"{len(trades)} trades"
                )

        trades_df = pd.DataFrame(
            all_trades
        )

        print(
            "\n=== V1.3 Relative Strength ==="
        )

        if trades_df.empty:
            print("No trades.")
            return

        returns = pd.to_numeric(
            trades_df["return"],
            errors="coerce",
        ).dropna()

        alpha = pd.to_numeric(
            trades_df["alpha"],
            errors="coerce",
        ).dropna()

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        gross_profit = float(
            wins.sum()
        )

        gross_loss = abs(
            float(losses.sum())
        )

        print("\n[Performance]")

        print(
            f"Trades: {len(trades_df)}"
        )

        print(
            f"Win Rate: "
            f"{(returns > 0).mean() * 100:.2f}%"
        )

        print(
            f"Avg Return: "
            f"{returns.mean() * 100:.2f}%"
        )

        print(
            f"Avg Alpha: "
            f"{alpha.mean() * 100:.2f}%"
        )

        print(
            f"Median Alpha: "
            f"{alpha.median() * 100:.2f}%"
        )

        print(
            f"Alpha > 0: "
            f"{(alpha > 0).mean() * 100:.2f}%"
        )

        print(
            f"Profit Factor: "
            f"{gross_profit / gross_loss:.2f}"
            if gross_loss > 0
            else "Profit Factor: 0"
        )

        # ----------------------------------------------
        # Relative Strength buckets
        # ----------------------------------------------

        bins = [
            float("-inf"),
            -0.10,
            -0.05,
            0,
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
            trades_df["rs20"],
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
                trades=("return", "count"),
                avg_return=(
                    "return",
                    "mean",
                ),
                avg_alpha=(
                    "alpha",
                    "mean",
                ),
                win_rate=(
                    "return",
                    lambda x: (
                        x > 0
                    ).mean(),
                ),
            )
        )

        rs_analysis[
            "avg_return"
        ] *= 100

        rs_analysis[
            "avg_alpha"
        ] *= 100

        rs_analysis[
            "win_rate"
        ] *= 100

        print(
            "\n[RS20 Analysis]"
        )

        print(
            rs_analysis.round(2)
        )

        # ----------------------------------------------
        # Regime analysis
        # ----------------------------------------------

        regime_analysis = (
            trades_df
            .groupby(
                "market_regime"
            )
            .agg(
                trades=("return", "count"),
                avg_return=(
                    "return",
                    "mean",
                ),
                avg_alpha=(
                    "alpha",
                    "mean",
                ),
                win_rate=(
                    "return",
                    lambda x: (
                        x > 0
                    ).mean(),
                ),
            )
        )

        regime_analysis[
            "avg_return"
        ] *= 100

        regime_analysis[
            "avg_alpha"
        ] *= 100

        regime_analysis[
            "win_rate"
        ] *= 100

        print(
            "\n[Market Regime Analysis]"
        )

        print(
            regime_analysis.round(2)
        )

        # ----------------------------------------------
        # Save
        # ----------------------------------------------

        trades_df.to_csv(
            "trades_relative_strength_v1_3.csv",
            index=False,
        )

        rs_analysis.to_csv(
            "rs_analysis_v1_3.csv"
        )

        regime_analysis.to_csv(
            "regime_analysis_v1_3.csv"
        )

        print(
            "\nSaved V1.3 reports."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
