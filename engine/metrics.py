from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, digits: int = 4) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), digits)


def _max_drawdown(returns: pd.Series) -> float:
    """
    Equal-weight trade equity curve.
    每筆交易視為 1 單位資金，避免現階段混入倉位配置假設。
    """
    if returns.empty:
        return 0.0

    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    return float(drawdown.min())


def _sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0

    std = returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return 0.0

    return float(returns.mean() / std * math.sqrt(len(returns)))


def _score_bucket(score: float) -> str:
    if score < 70:
        return "<70"
    if score < 75:
        return "70-75"
    if score < 80:
        return "75-80"
    if score < 85:
        return "80-85"
    if score < 90:
        return "85-90"
    return "90+"


def evaluate_backtest(trades: list[dict] | pd.DataFrame) -> dict[str, Any]:
    """
    產生完整的 baseline backtest metrics。

    注意：
    - 目前把每筆 trade 視為等權重，不假設實際資金配置。
    - 不把交易成本偷偷加進去，避免改變目前 baseline。
    """
    if isinstance(trades, pd.DataFrame):
        df = trades.copy()
    else:
        df = pd.DataFrame(trades)

    if df.empty:
        return {
            "performance": {
                "trades": 0,
                "win_rate": 0.0,
                "avg_return_pct": 0.0,
                "median_return_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
                "total_return_pct": 0.0,
            },
            "score_analysis": {},
            "exit_analysis": {},
            "stock_analysis": {},
        }

    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    df = df.dropna(subset=["return"]).copy()

    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(
            df["entry_date"],
            errors="coerce",
        )

    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(
            df["exit_date"],
            errors="coerce",
        )

    if "holding_days" not in df.columns:
        if "entry_date" in df.columns and "exit_date" in df.columns:
            holding = (
                df["exit_date"] - df["entry_date"]
            ).dt.days + 1
            df["holding_days"] = holding.fillna(0).astype(int)

    returns = df["return"].astype(float)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = float((returns > 0).mean())
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    # 每筆交易視為 1 單位資金。
    total_return = float((1.0 + returns).prod() - 1.0)

    performance = {
        "trades": int(len(df)),
        "win_rate_pct": _safe_float(win_rate * 100, 2),
        "avg_return_pct": _safe_float(returns.mean() * 100, 2),
        "median_return_pct": _safe_float(returns.median() * 100, 2),
        "avg_win_pct": _safe_float(wins.mean() * 100 if not wins.empty else 0, 2),
        "avg_loss_pct": _safe_float(losses.mean() * 100 if not losses.empty else 0, 2),
        "profit_factor": _safe_float(
            gross_profit / gross_loss if gross_loss > 0 else 0,
            2,
        ),
        "expectancy_pct": _safe_float(
            returns.mean() * 100,
            2,
        ),
        "total_return_pct": _safe_float(total_return * 100, 2),
        "max_drawdown_pct": _safe_float(_max_drawdown(returns) * 100, 2),
        "sharpe": _safe_float(_sharpe_ratio(returns), 2),
    }

    if "holding_days" in df.columns:
        performance["avg_holding_days"] = _safe_float(
            df["holding_days"].mean(),
            2,
        )
        performance["median_holding_days"] = _safe_float(
            df["holding_days"].median(),
            2,
        )

    # Exit analysis
    exit_analysis: dict[str, dict[str, Any]] = {}

    if "exit_reason" in df.columns:
        grouped = df.groupby("exit_reason", dropna=False)

        for reason, group in grouped:
            group_returns = group["return"].astype(float)

            exit_analysis[str(reason)] = {
                "trades": int(len(group)),
                "share_pct": _safe_float(len(group) / len(df) * 100, 2),
                "win_rate_pct": _safe_float(
                    (group_returns > 0).mean() * 100,
                    2,
                ),
                "avg_return_pct": _safe_float(
                    group_returns.mean() * 100,
                    2,
                ),
            }

    # Score bucket analysis
    score_analysis: dict[str, dict[str, Any]] = {}

    if "score" in df.columns:
        df["score_bucket"] = df["score"].apply(_score_bucket)

        bucket_order = [
            "<70",
            "70-75",
            "75-80",
            "80-85",
            "85-90",
            "90+",
        ]

        for bucket in bucket_order:
            group = df[df["score_bucket"] == bucket]

            if group.empty:
                continue

            group_returns = group["return"].astype(float)

            score_analysis[bucket] = {
                "trades": int(len(group)),
                "win_rate_pct": _safe_float(
                    (group_returns > 0).mean() * 100,
                    2,
                ),
                "avg_return_pct": _safe_float(
                    group_returns.mean() * 100,
                    2,
                ),
                "median_return_pct": _safe_float(
                    group_returns.median() * 100,
                    2,
                ),
            }

    # Stock analysis
    stock_analysis: dict[str, dict[str, Any]] = {}

    if "stock_id" in df.columns:
        grouped = df.groupby("stock_id")

        for stock_id, group in grouped:
            group_returns = group["return"].astype(float)

            stock_analysis[str(stock_id)] = {
                "trades": int(len(group)),
                "win_rate_pct": _safe_float(
                    (group_returns > 0).mean() * 100,
                    2,
                ),
                "avg_return_pct": _safe_float(
                    group_returns.mean() * 100,
                    2,
                ),
                "total_return_pct": _safe_float(
                    ((1.0 + group_returns).prod() - 1.0) * 100,
                    2,
                ),
            }

    return {
        "performance": performance,
        "score_analysis": score_analysis,
        "exit_analysis": exit_analysis,
        "stock_analysis": stock_analysis,
    }


def summarize_trades(trades: list[dict]) -> dict[str, Any]:
    """
    Backward-compatible wrapper。
    """
    result = evaluate_backtest(trades)
    return result["performance"]