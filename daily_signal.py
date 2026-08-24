from __future__ import annotations

"""Generate the daily V1.5 live-trading signal sheet.

This module intentionally does not place orders. It converts the frozen V1.5
research rules into a repeatable daily candidate scan.

Parity rules:
- Market regime is used for portfolio-level exits, not as an entry gate.
- Candidate ranking is RS20 descending, then score descending, then stock_id.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from engine.market_regime import build_market_proxy
from features.relative_strength import (
    add_relative_strength_features,
    build_relative_strength_signal,
)

DB_PATH = "data/database.db"
DEFAULT_OUTPUT = "reports/daily_signal.csv"

MAX_POSITIONS = 3
RS_THRESHOLD = 0.10
PULLBACK_MIN = -0.07
PULLBACK_MAX = -0.02
REGIME_POLICY = "bull_to_neutral"
MIN_SCORE = 70.0


def load_stock_data(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Load active stocks from daily_price, excluding quarantined tickers."""
    query = """
        SELECT p.stock_id, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_price p
        LEFT JOIN price_update_status s ON s.stock_id = p.stock_id
        WHERE COALESCE(s.quarantined, 0) = 0
        ORDER BY p.date, p.stock_id
    """
    raw = pd.read_sql_query(query, conn)
    if raw.empty:
        return {}

    raw["stock_id"] = raw["stock_id"].astype(str)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["stock_id", "date", "close"])
    raw = raw[raw["close"] > 0].copy()

    result: dict[str, pd.DataFrame] = {}
    for stock_id, group in raw.groupby("stock_id", sort=True):
        data = group.sort_values("date").set_index("date")
        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        result[stock_id] = data

    return result


def is_v15_candidate(
    *,
    rs20: float,
    drawdown_20d: float,
    score: float,
    trend_pass: bool,
    momentum_pass: bool,
) -> bool:
    """Apply the frozen V1.5 entry candidate definition."""
    return (
        rs20 > RS_THRESHOLD
        and PULLBACK_MIN <= drawdown_20d <= PULLBACK_MAX
        and score >= MIN_SCORE
        and trend_pass
        and momentum_pass
    )


def rank_and_select_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen V1.5 ranking and position cap to a signal table."""
    if signals.empty:
        result = signals.copy()
        result["selected"] = pd.Series(dtype=bool)
        result["action"] = pd.Series(dtype=str)
        return result

    result = signals.sort_values(
        ["candidate", "rs20", "score", "stock_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    candidate_idx = result.index[result["candidate"]].tolist()
    selected_idx = set(candidate_idx[:MAX_POSITIONS])
    result["selected"] = result.index.isin(selected_idx)
    result["action"] = result["selected"].map(
        {True: "BUY_NEXT_OPEN", False: "WATCH"}
    )
    return result


def build_signal_sheet(stock_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    """Calculate V1.5 signals for the latest common market date."""
    if not stock_data:
        raise ValueError("No stock data found in daily_price.")

    market = build_market_proxy(stock_data)
    if market.empty:
        raise ValueError("Unable to build market proxy from daily_price.")

    latest_date = market.index.max()
    latest_regime = str(market.loc[latest_date, "regime"])

    rows: list[dict] = []
    for stock_id, df in stock_data.items():
        if len(df) < 61 or latest_date not in df.index:
            continue

        data = add_relative_strength_features(df, market)
        if latest_date not in data.index:
            continue

        row = data.loc[latest_date]
        signal = build_relative_strength_signal(row)

        rs20 = float(row.get("rs20", 0.0))
        drawdown = float(row.get("drawdown_20d", 0.0))
        score = float(signal["score"])
        trend_pass = bool(signal["trend_pass"])
        momentum_pass = bool(signal["momentum_pass"])

        candidate = is_v15_candidate(
            rs20=rs20,
            drawdown_20d=drawdown,
            score=score,
            trend_pass=trend_pass,
            momentum_pass=momentum_pass,
        )

        rows.append(
            {
                "date": latest_date.date().isoformat(),
                "stock_id": stock_id,
                "market_regime": latest_regime,
                "close": float(row["Close"]),
                "rs20": rs20,
                "rs60": float(row.get("rs60", 0.0)),
                "drawdown_20d": drawdown,
                "score": score,
                "trend_pass": trend_pass,
                "momentum_pass": momentum_pass,
                "pullback_pass": bool(signal["pullback_pass"]),
                "candidate": candidate,
                "reason": "candidate" if candidate else "does_not_meet_v1_5_rules",
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        return signals, latest_regime

    return rank_and_select_signals(signals), latest_regime


def run(db_path: str = DB_PATH, output_path: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        stock_data = load_stock_data(conn)

    signals, regime = build_signal_sheet(stock_data)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output, index=False)

    selected = signals[signals["selected"]] if not signals.empty else signals
    print(f"Signal date: {signals['date'].iloc[0] if not signals.empty else 'N/A'}")
    print(f"Market regime: {regime}")
    print(f"Selected: {len(selected)} / {MAX_POSITIONS}")
    if not selected.empty:
        print(selected[["stock_id", "score", "rs20", "drawdown_20d", "action"]].to_string(index=False))
    print(f"Saved: {output}")

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V1.5 daily live signals")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Signal CSV output path")
    args = parser.parse_args()
    run(db_path=args.db, output_path=args.output)


if __name__ == "__main__":
    main()
