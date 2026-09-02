from __future__ import annotations

"""Generate the daily V1.6 portfolio-aware live-trading signal sheet.

V1.5 remains the deterministic stock-selection layer. V1.6 adds only the
portfolio rules currently chosen for live use: minimum score, BEAR-market entry
block, and HOLD for stocks already in the user's portfolio. Total capital is
optional metadata and does not control BUY sizing or cash availability.
"""

import argparse
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from engine.market_regime import build_market_proxy
from engine.portfolio_engine import (
    DEFAULT_STATE_PATH,
    apply_portfolio_decisions,
    load_portfolio_state,
)
from features.relative_strength import (
    add_relative_strength_features,
    build_relative_strength_signal,
)

DB_PATH = "data/database.db"
DEFAULT_OUTPUT = "reports/daily_signal.csv"
DEFAULT_PORTFOLIO_STATE = str(DEFAULT_STATE_PATH)

RS_THRESHOLD = 0.10
PULLBACK_MIN = -0.07
PULLBACK_MAX = -0.02
MIN_SIGNAL_SCORE = 70.0
MAX_CANDIDATES = 10


def load_stock_data(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Load only active, non-quarantined stocks from the local database."""
    query = """
        SELECT p.stock_id, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_price p
        INNER JOIN stock_universe u
            ON u.stock_id = p.stock_id
           AND u.is_active = 1
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
    """Apply the frozen V1.5 technical candidate definition."""
    return (
        rs20 > RS_THRESHOLD
        and PULLBACK_MIN <= drawdown_20d <= PULLBACK_MAX
        and score >= MIN_SIGNAL_SCORE
        and trend_pass
        and momentum_pass
    )


def build_signal_sheet(
    stock_data: dict[str, pd.DataFrame],
    *,
    portfolio_state_path: str | Path = DEFAULT_PORTFOLIO_STATE,
) -> tuple[pd.DataFrame, str]:
    """Calculate V1.5 signals and apply the simplified V1.6 portfolio gates."""
    if not stock_data:
        raise ValueError("No active stock data found in daily_price.")

    market = build_market_proxy(stock_data)
    if market.empty:
        raise ValueError("Unable to build market proxy from daily_price.")

    latest_date = market.index.max()
    signal_date: date = latest_date.date()
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
                "date": signal_date.isoformat(),
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

    state = load_portfolio_state(portfolio_state_path)
    signals = apply_portfolio_decisions(
        signals,
        state,
        signal_date=signal_date,
        max_candidates=MAX_CANDIDATES,
    )
    return signals, latest_regime


def run(
    db_path: str = DB_PATH,
    output_path: str = DEFAULT_OUTPUT,
    portfolio_state_path: str = DEFAULT_PORTFOLIO_STATE,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        stock_data = load_stock_data(conn)

    signals, regime = build_signal_sheet(
        stock_data,
        portfolio_state_path=portfolio_state_path,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output, index=False)

    selected = signals[signals["selected"]] if not signals.empty else signals
    candidates = signals[signals["candidate"]] if not signals.empty else signals
    top_candidates = signals[signals["top_candidate"]] if not signals.empty else signals
    signal_date = signals["date"].iloc[0] if not signals.empty else "N/A"

    print(f"Signal date: {signal_date}")
    print(f"Market regime: {regime}")
    print(f"Universe scanned: {len(stock_data)}")
    print(f"V1.5 trade candidates: {len(candidates)}")
    print(f"Top {MAX_CANDIDATES} candidates shown: {len(top_candidates)}")
    print(f"Selected BUYs after V1.6 gates: {len(selected)}")
    if not selected.empty:
        print(
            selected[
                ["stock_id", "score", "rs20", "drawdown_20d", "action", "portfolio_reason"]
            ].to_string(index=False)
        )
    else:
        print("No new BUY selected today.")
    print(f"Saved: {output}")
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V1.6 daily portfolio-aware signals")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Signal CSV output path")
    parser.add_argument(
        "--portfolio-state",
        default=DEFAULT_PORTFOLIO_STATE,
        help="JSON file containing optional total capital and current holdings",
    )
    args = parser.parse_args()
    run(
        db_path=args.db,
        output_path=args.output,
        portfolio_state_path=args.portfolio_state,
    )


if __name__ == "__main__":
    main()
