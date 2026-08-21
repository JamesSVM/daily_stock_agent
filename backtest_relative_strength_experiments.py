from __future__ import annotations

import pandas as pd


TRADES_FILE = "trades_relative_strength_v1_3.csv"


def calculate_metrics(
    df: pd.DataFrame,
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
            "max_drawdown_pct": 0.0,
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

    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0

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
        "max_drawdown_pct": round(
            float(drawdown.min() * 100),
            2,
        ),
    }


def apply_daily_top_n(
    df: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    data = df.copy()

    data["entry_date"] = pd.to_datetime(
        data["entry_date"],
        errors="coerce",
    )

    data = data.sort_values(
        [
            "entry_date",
            "rs20",
            "score",
            "stock_id",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    return (
        data
        .groupby(
            "entry_date",
            group_keys=False,
        )
        .head(n)
        .copy()
    )


def main() -> None:
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

    trades["score"] = pd.to_numeric(
        trades["score"],
        errors="coerce",
    )

    experiments: list[tuple[str, pd.DataFrame]] = []

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    experiments.append(
        (
            "baseline",
            trades.copy(),
        )
    )

    # --------------------------------------------------
    # RS threshold
    # --------------------------------------------------

    experiments.append(
        (
            "rs20_gt_5",
            trades[
                trades["rs20"] > 0.05
            ].copy(),
        )
    )

    experiments.append(
        (
            "rs20_gt_10",
            trades[
                trades["rs20"] > 0.10
            ].copy(),
        )
    )

    # --------------------------------------------------
    # Pullback / Drawdown
    # --------------------------------------------------

    experiments.append(
        (
            "pullback_only",
            trades[
                (trades["drawdown_20d"] <= -0.02)
                & (trades["drawdown_20d"] >= -0.10)
            ].copy(),
        )
    )

    # --------------------------------------------------
    # RS + Pullback
    # --------------------------------------------------

    experiments.append(
        (
            "rs10_pullback",
            trades[
                (trades["rs20"] > 0.10)
                & (trades["drawdown_20d"] <= -0.02)
                & (trades["drawdown_20d"] >= -0.10)
            ].copy(),
        )
    )

    experiments.append(
        (
            "rs5_pullback",
            trades[
                (trades["rs20"] > 0.05)
                & (trades["drawdown_20d"] <= -0.02)
                & (trades["drawdown_20d"] >= -0.10)
            ].copy(),
        )
    )

    # --------------------------------------------------
    # Stronger pullback
    # --------------------------------------------------

    experiments.append(
        (
            "rs10_pullback_2_7",
            trades[
                (trades["rs20"] > 0.10)
                & (trades["drawdown_20d"] <= -0.02)
                & (trades["drawdown_20d"] >= -0.07)
            ].copy(),
        )
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    rows = []

    for name, data in experiments:

        metrics = calculate_metrics(
            data,
        )

        metrics["experiment"] = name

        rows.append(
            metrics
        )

    results = pd.DataFrame(rows)

    results = results[
        [
            "experiment",
            "trades",
            "win_rate_pct",
            "avg_return_pct",
            "avg_alpha_pct",
            "median_alpha_pct",
            "alpha_positive_pct",
            "profit_factor",
            "max_drawdown_pct",
        ]
    ]

    print(
        "\n=== V1.3-B Selectivity Experiments ==="
    )

    print(
        results.to_string(
            index=False,
        )
    )

    results.to_csv(
        "relative_strength_selectivity_v1_3.csv",
        index=False,
    )

    print(
        "\nSaved: "
        "relative_strength_selectivity_v1_3.csv"
    )


if __name__ == "__main__":
    main()
