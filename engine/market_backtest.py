from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from engine.backtest_mean_reversion import simulate_stock
from features.mean_reversion import add_mean_reversion_features, build_signal


def load_stock_ids(db_path: str = "data/database.db") -> list[str]:
    """Return all stock IDs currently available in daily_price."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT stock_id FROM daily_price ORDER BY stock_id"
        ).fetchall()

    return [str(row[0]) for row in rows]


def load_stock_data(
    stock_id: str,
    db_path: str = "data/database.db",
) -> pd.DataFrame:
    """Load one stock's OHLCV data from SQLite in chronological order."""
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

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[stock_id])

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")

    return df


def scan_signal_diagnostics(
    df: pd.DataFrame,
    stock_id: str,
    min_score: float = 70.0,
) -> dict:
    """Summarize how often the raw signal conditions pass or fail."""
    data = add_mean_reversion_features(df).copy()
    data["prev_rsi14"] = data["rsi14"].shift(1)
    data = data.dropna(subset=["ma60", "rsi14", "atr14"])

    rows = []
    for _, row in data.iterrows():
        signal = build_signal(row)
        score = float(signal["score"])
        rows.append(
            {
                "stock_id": stock_id,
                "score": score,
                "score_pass": score >= min_score,
                "reversal_pass": bool(signal["reversal_pass"]),
                "fundamental_pass": bool(signal["fundamental_pass"]),
                "buy_zone": bool(signal["buy_zone"]),
                "oversold_score": float(signal["oversold_score"]),
                "reversal_score": float(signal["reversal_score"]),
            }
        )

    if not rows:
        return {
            "stock_id": stock_id,
            "eligible_days": 0,
            "score_pass_days": 0,
            "reversal_pass_days": 0,
            "buy_zone_days": 0,
            "score_pass_rate_pct": 0.0,
            "reversal_pass_rate_pct": 0.0,
            "buy_zone_rate_pct": 0.0,
            "avg_score": 0.0,
        }

    diag = pd.DataFrame(rows)
    n = len(diag)

    return {
        "stock_id": stock_id,
        "eligible_days": n,
        "score_pass_days": int(diag["score_pass"].sum()),
        "reversal_pass_days": int(diag["reversal_pass"].sum()),
        "buy_zone_days": int(diag["buy_zone"].sum()),
        "score_pass_rate_pct": round(float(diag["score_pass"].mean() * 100), 2),
        "reversal_pass_rate_pct": round(float(diag["reversal_pass"].mean() * 100), 2),
        "buy_zone_rate_pct": round(float(diag["buy_zone"].mean() * 100), 2),
        "avg_score": round(float(diag["score"].mean()), 2),
    }


def run_market_backtest(
    stock_ids: Iterable[str],
    db_path: str = "data/database.db",
    hold_days: int = 10,
    stop_loss: float = 0.05,
    take_profit: float = 0.10,
    min_score: float = 70.0,
    min_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the unchanged single-stock strategy over every stock in the DB."""
    all_trades: list[dict] = []
    diagnostics: list[dict] = []

    for idx, stock_id in enumerate(stock_ids, start=1):
        df = load_stock_data(stock_id, db_path=db_path)
        if len(df) < min_rows:
            diagnostics.append(
                {
                    "stock_id": stock_id,
                    "eligible_days": len(df),
                    "score_pass_days": 0,
                    "reversal_pass_days": 0,
                    "buy_zone_days": 0,
                    "score_pass_rate_pct": 0.0,
                    "reversal_pass_rate_pct": 0.0,
                    "buy_zone_rate_pct": 0.0,
                    "avg_score": 0.0,
                    "status": "insufficient_data",
                    "trades": 0,
                }
            )
            continue

        diag = scan_signal_diagnostics(
            df,
            stock_id=stock_id,
            min_score=min_score,
        )

        trades = simulate_stock(
            df,
            stock_id=stock_id,
            hold_days=hold_days,
            stop_loss=stop_loss,
            take_profit=take_profit,
            min_score=min_score,
        )

        all_trades.extend(trades)
        diag["status"] = "ok"
        diag["trades"] = len(trades)
        diagnostics.append(diag)

        print(f"[{idx:03d}/{len(list(stock_ids))}] {stock_id}: {len(trades)} trades")

    trades_df = pd.DataFrame(all_trades)
    diagnostics_df = pd.DataFrame(diagnostics)

    if not trades_df.empty:
        trades_df["signal_date"] = pd.to_datetime(
            trades_df["signal_date"], errors="coerce"
        )
        trades_df["entry_date"] = pd.to_datetime(
            trades_df["entry_date"], errors="coerce"
        )
        trades_df["exit_date"] = pd.to_datetime(
            trades_df["exit_date"], errors="coerce"
        )
        trades_df["holding_days"] = (
            trades_df["exit_date"] - trades_df["entry_date"]
        ).dt.days + 1

        trades_df = trades_df.sort_values(
            ["entry_date", "stock_id"]
        ).reset_index(drop=True)

    return trades_df, diagnostics_df


def summarize_trades(trades_df: pd.DataFrame) -> dict:
    """Trade-level summary for cross-sectional strategy research."""
    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "trade_compound_return_pct": 0.0,
            "max_trade_drawdown_pct": 0.0,
            "avg_holding_days": 0.0,
        }

    returns = pd.to_numeric(trades_df["return"], errors="coerce").dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    return {
        "trades": int(len(returns)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
        "avg_return_pct": round(float(returns.mean() * 100), 2),
        "median_return_pct": round(float(returns.median() * 100), 2),
        "avg_win_pct": round(float(wins.mean() * 100) if not wins.empty else 0.0, 2),
        "avg_loss_pct": round(float(losses.mean() * 100) if not losses.empty else 0.0, 2),
        "profit_factor": round(
            gross_profit / gross_loss if gross_loss > 0 else 0.0,
            2,
        ),
        "trade_compound_return_pct": round(
            float((1.0 + returns).prod() - 1.0) * 100,
            2,
        ),
        "max_trade_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "avg_holding_days": round(
            float(trades_df["holding_days"].mean()),
            2,
        ),
    }


def build_analysis_tables(trades_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create score, exit, and stock cross-sections for V1.1 analysis."""
    if trades_df.empty:
        return {
            "score_analysis": pd.DataFrame(),
            "exit_analysis": pd.DataFrame(),
            "stock_analysis": pd.DataFrame(),
        }

    df = trades_df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")

    bins = [-float("inf"), 70, 75, 80, 85, 90, float("inf")]
    labels = ["<70", "70-75", "75-80", "80-85", "85-90", "90+"]
    df["score_bucket"] = pd.cut(
        df["score"], bins=bins, labels=labels, right=False
    )

    score = (
        df.groupby("score_bucket", observed=False)
        .agg(
            trades=("return", "count"),
            win_rate_pct=("return", lambda s: float((s > 0).mean() * 100)),
            avg_return_pct=("return", lambda s: float(s.mean() * 100)),
            median_return_pct=("return", lambda s: float(s.median() * 100)),
        )
        .reset_index()
    )

    exits = (
        df.groupby("exit_reason")
        .agg(
            trades=("return", "count"),
            win_rate_pct=("return", lambda s: float((s > 0).mean() * 100)),
            avg_return_pct=("return", lambda s: float(s.mean() * 100)),
        )
        .reset_index()
    )
    exits["share_pct"] = exits["trades"] / len(df) * 100

    stocks = (
        df.groupby("stock_id")
        .agg(
            trades=("return", "count"),
            win_rate_pct=("return", lambda s: float((s > 0).mean() * 100)),
            avg_return_pct=("return", lambda s: float(s.mean() * 100)),
            median_return_pct=("return", lambda s: float(s.median() * 100)),
        )
        .reset_index()
        .sort_values("avg_return_pct", ascending=False)
    )

    for table in (score, exits, stocks):
        for column in table.columns:
            if column.endswith("_pct"):
                table[column] = table[column].round(2)

    return {
        "score_analysis": score,
        "exit_analysis": exits,
        "stock_analysis": stocks,
    }


def save_reports(
    output_dir: str,
    trades_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    summary: dict,
    analysis_tables: dict[str, pd.DataFrame],
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_csv(path / "trades_mean_reversion_v11.csv", index=False)

    diagnostics_df.to_csv(path / "signal_diagnostics_v11.csv", index=False)
    pd.DataFrame([summary]).to_csv(path / "backtest_summary_v11.csv", index=False)

    for name, table in analysis_tables.items():
        table.to_csv(path / f"{name}_v11.csv", index=False)
