from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from engine.market_regime import build_market_proxy
from engine.performance_tracker import (
    build_signal_events,
    rank_daily_signals,
    summarize_bucket_performance,
    summarize_persistence,
)
from features.relative_strength import add_relative_strength_features, build_relative_strength_signal

DEFAULT_DB = REPO_ROOT / "data" / "database.db"
REPORT_DIR = REPO_ROOT / "reports" / "performance"


def _load_universe_stock_data(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    query = """
        SELECT p.stock_id, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM daily_price p
        INNER JOIN stock_universe u ON u.stock_id = p.stock_id AND u.is_active = 1
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
    raw = raw[raw["close"] > 0]

    result: dict[str, pd.DataFrame] = {}
    for stock_id, group in raw.groupby("stock_id", sort=True):
        result[stock_id] = group.sort_values("date").set_index("date").rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )
    return result


def _build_daily_rankings(stock_data: dict[str, pd.DataFrame]) -> list[pd.DataFrame]:
    market = build_market_proxy(stock_data)
    if market.empty:
        return []

    by_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
    for stock_id, df in stock_data.items():
        if len(df) < 61:
            continue
        features = add_relative_strength_features(df, market)
        for signal_date, row in features.iterrows():
            if pd.isna(signal_date):
                continue
            signal = build_relative_strength_signal(row)
            if any(pd.isna(row.get(col)) for col in ["rs20", "rs60", "drawdown_20d"]):
                continue
            by_date.setdefault(signal_date, []).append({
                "date": signal_date.date().isoformat(),
                "stock_id": stock_id,
                "market_regime": str(market.loc[signal_date, "regime"]) if signal_date in market.index else "neutral",
                "close": float(row["Close"]),
                "rs20": float(row["rs20"]),
                "rs60": float(row["rs60"]),
                "drawdown_20d": float(row["drawdown_20d"]),
                "score": float(signal["score"]),
                "trend_pass": bool(signal["trend_pass"]),
                "momentum_pass": bool(signal["momentum_pass"]),
                "pullback_pass": bool(signal["pullback_pass"]),
                "candidate": bool(
                    signal["score"] >= 70.0
                    and signal["rs20"] > 0.10
                    and -0.07 <= signal["drawdown_20d"] <= -0.02
                    and signal["trend_pass"]
                    and signal["momentum_pass"]
                ),
            })

    return [rank_daily_signals(pd.DataFrame(by_date[d]), top_n=10) for d in sorted(by_date)]


def _load_taiwan_benchmark(period: str = "3mo") -> pd.DataFrame:
    try:
        from crawler.price import get_price
        benchmark = get_price("^TWII", period=period)
        if benchmark is None or benchmark.empty:
            return pd.DataFrame()
        data = benchmark.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={"Date": "date", "Open": "open", "Close": "close"})
        return data[["date", "open", "close"]]
    except Exception as exc:  # benchmark is supplemental
        print(f"Benchmark warning: TAIEX unavailable ({exc})")
        return pd.DataFrame()


def _price_frames_for_tracker(stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for stock_id, data in stock_data.items():
        if data is None or data.empty:
            continue
        frame = data.reset_index()
        if "date" not in frame.columns and "Date" in frame.columns:
            frame = frame.rename(columns={"Date": "date"})
        frames[stock_id] = frame
    return frames


def run(db_path: str, benchmark_period: str = "3mo") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        stock_data = _load_universe_stock_data(conn)
    if not stock_data:
        raise RuntimeError("No active stock data found. Refresh the 300-stock universe first.")

    rankings = _build_daily_rankings(stock_data)
    benchmark = _load_taiwan_benchmark(benchmark_period)
    events = build_signal_events(rankings, _price_frames_for_tracker(stock_data), benchmark=benchmark)
    summary = summarize_bucket_performance(events)
    persistence = summarize_persistence(events)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(REPORT_DIR / "signal_events_top10.csv", index=False)
    summary.to_csv(REPORT_DIR / "performance_summary.csv", index=False)
    persistence.to_csv(REPORT_DIR / "persistence_summary.csv", index=False)

    print(f"Universe scanned: {len(stock_data)}")
    print(f"Signal dates: {len(rankings)}")
    print(f"Top-10 signal events: {len(events)}")
    print(f"Saved: {REPORT_DIR / 'signal_events_top10.csv'}")
    print(f"Saved: {REPORT_DIR / 'performance_summary.csv'}")
    print(f"Saved: {REPORT_DIR / 'persistence_summary.csv'}")
    return events, summary, persistence


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V1.6 signal performance reports")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--benchmark-period", default="3mo")
    args = parser.parse_args()
    run(args.db, benchmark_period=args.benchmark_period)


if __name__ == "__main__":
    main()
