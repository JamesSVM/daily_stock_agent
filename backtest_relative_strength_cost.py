from __future__ import annotations

import pandas as pd


TRADES_FILE = "trades_relative_strength_v1_3.csv"


def calculate_metrics(
    df: pd.DataFrame,
    cost: float = 0.0,
) -> dict:

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

    data = df.copy()

    data["return"] = pd.to_numeric(
        data["return"],
        errors="coerce",
    )

    data["market_return"] = pd.to_numeric(
        data["market_return"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "return",
            "market_return",
        ]
    )

    # --------------------------------------------------
    # Transaction cost
    #
    # cost = total round-trip cost
    # e.g. 0.003 = 0.3%
    # --------------------------------------------------

    data["net_return"] = (
        (1.0 + data["return"])
        * (1.0 - cost)
        - 1.0
    )

    # Alpha against the same market benchmark.
    data["net_alpha"] = (
        data["net_return"]
        - data["market_return"]
    )

    returns = data["net_return"]

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = float(
        wins.sum()
    )

    gross_loss = abs(
        float(losses.sum())
    )

    return {
        "trades": int(len(data)),

        "win_rate_pct": round(
            float(
                (returns > 0).mean()
                * 100
            ),
            2,
        ),

        "avg_return_pct": round(
            float(
                returns.mean()
                * 100
            ),
            2,
        ),

        "avg_alpha_pct": round(
            float(
                data["net_alpha"].mean()
                * 100
            ),
            2,
        ),

        "median_alpha_pct": round(
            float(
                data["net_alpha"].median()
                * 100
            ),
            2,
        ),

        "alpha_positive_pct": round(
            float(
                (
                    data["net_alpha"] > 0
                ).mean()
                * 100
            ),
            2,
        ),

        "profit_factor": round(
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0,
            2,
        ),
    }


def run_period(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:

    data = df[
        (df["entry_date"].dt.year >= start_year)
        & (df["entry_date"].dt.year <= end_year)
    ].copy()

    rows = []

    for cost in [
        0.0,
        0.003,
        0.005,
        0.008,
    ]:

        metrics = calculate_metrics(
            data,
            cost=cost,
        )

        metrics["cost_pct"] = (
            cost * 100
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


def main():

    trades = pd.read_csv(
        TRADES_FILE,
    )

    trades["entry_date"] = pd.to_datetime(
        trades["entry_date"],
        errors="coerce",
    )

    trades["rs20"] = pd.to_numeric(
        trades["rs20"],
        errors="coerce",
    )

    trades["drawdown_20d"] = pd.to_numeric(
        trades["drawdown_20d"],
        errors="coerce",
    )

    trades = trades[
        (trades["rs20"] > 0.10)
        & (trades["drawdown_20d"] >= -0.07)
        & (trades["drawdown_20d"] <= -0.02)
    ].copy()

    print(
        f"\nFixed V1.3-C setup trades: {len(trades)}"
    )

    print(
        "\nV1.3-C setup:"
        "\n  RS20 > 10%"
        "\n  Pullback = -7% ~ -2%"
    )

    print(
        "\n=== V1.3-D Transaction Cost Validation ==="
    )

    # --------------------------------------------------
    # Research
    # --------------------------------------------------

    research = run_period(
        trades,
        2021,
        2024,
    )

    print(
        "\n[Research 2021-2024]"
    )

    print(
        research.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Out-of-sample
    # --------------------------------------------------

    oos = run_period(
        trades,
        2025,
        2026,
    )

    print(
        "\n[Out-of-Sample 2025-2026]"
    )

    print(
        oos.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    result = pd.concat(
        [
            research.assign(
                period="research"
            ),
            oos.assign(
                period="oos"
            ),
        ],
        ignore_index=True,
    )

    result.to_csv(
        "relative_strength_cost_v1_3.csv",
        index=False,
    )

    print(
        "\nSaved: relative_strength_cost_v1_3.csv"
    )


if __name__ == "__main__":
    main()
