from __future__ import annotations

import sqlite3

import pandas as pd

from crawler.price import get_price
from engine.backtest_relative_strength import (
    simulate_stock_relative_strength,
)
from engine.market_regime import build_market_proxy


DB_PATH = "data/database.db"


# ------------------------------------------------------
# V1.3 fixed research setup
#
# We DO NOT optimize these parameters here.
# They were selected before this yearly analysis.
# ------------------------------------------------------

HOLD_DAYS = 10
STOP_LOSS = 0.05
TAKE_PROFIT = 0.10

MIN_SCORE = 70.0

ALLOWED_REGIMES = {
    "bull",
    "neutral",
}

RS_THRESHOLD = 0.10

PULLBACK_MIN = -0.07
PULLBACK_MAX = -0.02


def get_stock_ids(conn) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT stock_id
        FROM daily_price
        ORDER BY stock_id
        """
    ).fetchall()

    return [str(row[0]) for row in rows]


def calculate_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "avg_alpha_pct": 0.0,
            "median_alpha_pct": 0.0,
            "alpha_positive_pct": 0.0,
            "profit_factor": 0.0,
        }

    returns = pd.to_numeric(
        df["return"],
        errors="coerce",
    ).dropna()

    alpha = pd.to_numeric(
        df["alpha"],
        errors="coerce",
    ).dropna()

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    return {
        "trades": int(len(returns)),
        "win_rate_pct": round(
            float((returns > 0).mean() * 100),
            2,
        ),
        "avg_return_pct": round(
            float(returns.mean() * 100),
            2,
        ),
        "avg_alpha_pct": round(
            float(alpha.mean() * 100),
            2,
        ),
        "median_alpha_pct": round(
            float(alpha.median() * 100),
            2,
        ),
        "alpha_positive_pct": round(
            float((alpha > 0).mean() * 100),
            2,
        ),
        "profit_factor": round(
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0,
            2,
        ),
    }


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    try:
        stock_ids = get_stock_ids(conn)

        print(
            f"Universe: {len(stock_ids)} stocks"
        )

        # --------------------------------------------------
        # Load stock data
        # --------------------------------------------------

        stock_data: dict[str, pd.DataFrame] = {}

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

        # --------------------------------------------------
        # Build market proxy
        # --------------------------------------------------

        market = build_market_proxy(
            stock_data
        )

        if market.empty:
            print("Market proxy is empty.")
            return

        # --------------------------------------------------
        # Generate ALL V1.3 candidate trades
        # --------------------------------------------------

        all_trades: list[dict] = []

        for stock_id, df in stock_data.items():
            trades = simulate_stock_relative_strength(
                df,
                stock_id=stock_id,
                market=market,
                hold_days=HOLD_DAYS,
                stop_loss=STOP_LOSS,
                take_profit=TAKE_PROFIT,
                min_score=MIN_SCORE,
                allowed_regimes=ALLOWED_REGIMES,
            )

            all_trades.extend(trades)

        trades_df = pd.DataFrame(
            all_trades
        )

        if trades_df.empty:
            print("No trades generated.")
            return

        # --------------------------------------------------
        # Normalize dates / numeric fields
        # --------------------------------------------------

        trades_df["entry_date"] = pd.to_datetime(
            trades_df["entry_date"],
            errors="coerce",
        )

        trades_df["exit_date"] = pd.to_datetime(
            trades_df["exit_date"],
            errors="coerce",
        )

        trades_df["rs20"] = pd.to_numeric(
            trades_df["rs20"],
            errors="coerce",
        )

        trades_df["drawdown_20d"] = pd.to_numeric(
            trades_df["drawdown_20d"],
            errors="coerce",
        )

        trades_df["return"] = pd.to_numeric(
            trades_df["return"],
            errors="coerce",
        )

        trades_df["alpha"] = pd.to_numeric(
            trades_df["alpha"],
            errors="coerce",
        )

        trades_df = trades_df.dropna(
            subset=[
                "entry_date",
                "rs20",
                "drawdown_20d",
                "return",
                "alpha",
            ]
        )

        # --------------------------------------------------
        # V1.3-C fixed setup
        #
        # This is the setup we are validating:
        #
        # RS20 > +10%
        # AND
        # drawdown between -2% and -7%
        # --------------------------------------------------

        strategy = trades_df[
            (trades_df["rs20"] > RS_THRESHOLD)
            & (
                trades_df["drawdown_20d"]
                <= PULLBACK_MAX
            )
            & (
                trades_df["drawdown_20d"]
                >= PULLBACK_MIN
            )
        ].copy()

        # --------------------------------------------------
        # Year
        # --------------------------------------------------

        strategy["year"] = (
            strategy["entry_date"]
            .dt.year
        )

        rows = []

        for year, group in strategy.groupby(
            "year"
        ):
            metrics = calculate_metrics(
                group
            )

            metrics["year"] = int(year)

            rows.append(metrics)

        yearly = pd.DataFrame(rows)

        if not yearly.empty:
            yearly = yearly[
                [
                    "year",
                    "trades",
                    "win_rate_pct",
                    "avg_return_pct",
                    "avg_alpha_pct",
                    "median_alpha_pct",
                    "alpha_positive_pct",
                    "profit_factor",
                ]
            ].sort_values("year")

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        print(
            "\n=== V1.3-C Yearly Validation ==="
        )

        print(
            "\nFixed setup:"
        )

        print(
            f"RS20 > {RS_THRESHOLD * 100:.0f}%"
        )

        print(
            f"Pullback: "
            f"{PULLBACK_MIN * 100:.0f}% "
            f"to "
            f"{PULLBACK_MAX * 100:.0f}%"
        )

        print(
            "\n[Yearly Performance]"
        )

        print(
            yearly.to_string(
                index=False
            )
        )

        # --------------------------------------------------
        # Overall OOS-style split
        #
        # Research: <= 2024
        # OOS: >= 2025
        # --------------------------------------------------

        research = strategy[
            strategy["year"] <= 2024
        ]

        oos = strategy[
            strategy["year"] >= 2025
        ]

        research_metrics = calculate_metrics(
            research
        )

        oos_metrics = calculate_metrics(
            oos
        )

        print(
            "\n[Research Period: 2021-2024]"
        )

        for key, value in research_metrics.items():
            print(
                f"{key}: {value}"
            )

        print(
            "\n[Out-of-Sample: 2025-2026]"
        )

        for key, value in oos_metrics.items():
            print(
                f"{key}: {value}"
            )

        # --------------------------------------------------
        # Save reports
        # --------------------------------------------------

        strategy.to_csv(
            "trades_relative_strength_v1_3_validated.csv",
            index=False,
        )

        yearly.to_csv(
            "relative_strength_yearly_v1_3.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "period": "research_2021_2024",
                    **research_metrics,
                },
                {
                    "period": "oos_2025_2026",
                    **oos_metrics,
                },
            ]
        ).to_csv(
            "relative_strength_oos_v1_3.csv",
            index=False,
        )

        print(
            "\nSaved V1.3 validation reports."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
