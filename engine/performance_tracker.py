from __future__ import annotations

"""V1.6 signal-event and forward-performance tracking.

The tracker treats every daily rank as a separate signal event. The same stock
appearing on different dates therefore creates multiple events, while portfolio
simulation remains a separate layer that can enforce one live position per stock.

A signal is generated at T's close. The modeled entry is T+1 open. Forward
returns are measured from that entry open to the close on +1/+3/+5/+10 trading
days. No future rows are used to create the rank itself.
"""

from typing import Iterable

import pandas as pd

HORIZONS = (1, 3, 5, 10)
TOP_BUCKETS = (1, 3, 5, 10)


def rank_daily_signals(signals: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank a single-date signal table using the frozen V1.5 ordering."""
    if signals.empty:
        return signals.copy()

    required = {"stock_id", "score", "rs20", "candidate"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing signal columns: {sorted(missing)}")

    result = signals.copy()
    if "date" not in result.columns:
        raise ValueError("Signal table must contain date.")

    result = result.sort_values(
        ["candidate", "rs20", "score", "stock_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    result["rank"] = result.index + 1
    result["tracked"] = result["rank"] <= top_n
    return result


def top_bucket_membership(rank: int) -> tuple[str, ...]:
    """Return the cumulative Top-1/3/5/10 buckets containing a rank."""
    if rank > 10:
        return ()
    return tuple(f"top_{bucket}" for bucket in TOP_BUCKETS if rank <= bucket)


def _clean_series(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Date" in data.columns and "date" not in data.columns:
        data = data.rename(columns={"Date": "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    for col in ("open", "high", "low", "close", "volume"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def forward_returns(
    price_data: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> dict[str, object]:
    """Calculate entry-at-next-open and +1/+3/+5/+10 trading-day returns."""
    empty = {"entry_date": None, "entry_open": None, **{f"return_{h}d": None for h in HORIZONS}}
    if price_data is None or price_data.empty:
        return empty

    data = _clean_series(price_data)
    if data.empty or "open" not in data.columns or "close" not in data.columns:
        return empty

    data = data.reset_index(drop=True)
    dates = data["date"]
    after = data.index[dates > signal_date]
    if len(after) == 0:
        return empty

    entry_idx = int(after[0])
    entry_open = data.loc[entry_idx, "open"]
    if pd.isna(entry_open) or float(entry_open) <= 0:
        return empty

    result: dict[str, object] = {
        "entry_date": dates.iloc[entry_idx].date().isoformat(),
        "entry_open": float(entry_open),
    }
    for horizon in HORIZONS:
        target_idx = entry_idx + horizon - 1
        if target_idx >= len(data) or pd.isna(data.loc[target_idx, "close"]):
            result[f"return_{horizon}d"] = None
        else:
            result[f"return_{horizon}d"] = float(data.loc[target_idx, "close"] / entry_open - 1.0)
    return result


def build_signal_events(
    ranked_by_date: Iterable[pd.DataFrame],
    price_by_stock: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one event for every Top-10 stock on every signal date."""
    events: list[dict[str, object]] = []
    benchmark_data = _clean_series(benchmark) if benchmark is not None and not benchmark.empty else None

    for day_signals in ranked_by_date:
        if day_signals.empty:
            continue
        signal_date = pd.to_datetime(day_signals["date"].iloc[0], errors="coerce")
        if pd.isna(signal_date):
            continue

        tracked = day_signals[day_signals["rank"] <= 10]
        for _, row in tracked.iterrows():
            stock_id = str(row["stock_id"])
            returns = forward_returns(price_by_stock.get(stock_id, pd.DataFrame()), signal_date)
            benchmark_returns = forward_returns(benchmark_data, signal_date) if benchmark_data is not None else {}
            event = {
                "signal_date": signal_date.date().isoformat(),
                "stock_id": stock_id,
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "rs20": float(row.get("rs20", 0.0)),
                "rs60": float(row.get("rs60", 0.0)),
                "market_regime": str(row.get("market_regime", "neutral")),
                "close_price": float(row["close"]),
                "entry_date": returns.get("entry_date"),
                "entry_open": returns.get("entry_open"),
            }
            for horizon in HORIZONS:
                event[f"return_{horizon}d"] = returns.get(f"return_{horizon}d")
                event[f"benchmark_return_{horizon}d"] = benchmark_returns.get(f"return_{horizon}d")
                stock_ret = event[f"return_{horizon}d"]
                market_ret = event[f"benchmark_return_{horizon}d"]
                event[f"excess_return_{horizon}d"] = (
                    stock_ret - market_ret
                    if stock_ret is not None and market_ret is not None
                    else None
                )
            event["top_buckets"] = ",".join(top_bucket_membership(int(row["rank"])))
            events.append(event)

    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values(["signal_date", "rank", "stock_id"]).reset_index(drop=True)


def summarize_bucket_performance(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed forward returns for Top-1/3/5/10."""
    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()

    for bucket in TOP_BUCKETS:
        subset = events[events["rank"] <= bucket]
        for horizon in HORIZONS:
            col = f"return_{horizon}d"
            excess_col = f"excess_return_{horizon}d"
            values = pd.to_numeric(subset[col], errors="coerce").dropna()
            excess = pd.to_numeric(subset[excess_col], errors="coerce").dropna()
            if values.empty:
                rows.append({
                    "bucket": f"top_{bucket}",
                    "horizon": f"{horizon}d",
                    "samples": 0,
                    "win_rate": None,
                    "avg_return": None,
                    "median_return": None,
                    "avg_excess_return": None,
                })
                continue
            rows.append({
                "bucket": f"top_{bucket}",
                "horizon": f"{horizon}d",
                "samples": int(values.size),
                "win_rate": float((values > 0).mean()),
                "avg_return": float(values.mean()),
                "median_return": float(values.median()),
                "avg_excess_return": float(excess.mean()) if not excess.empty else None,
            })
    return pd.DataFrame(rows)


def _longest_consecutive_dates(dates: Iterable[str]) -> int:
    normalized = sorted({pd.to_datetime(value).normalize() for value in dates})
    if not normalized:
        return 0
    longest = current = 1
    for previous, current_date in zip(normalized, normalized[1:]):
        if (current_date - previous).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def summarize_persistence(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated appearances of the same stock in Top-10."""
    if events.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for stock_id, group in events.groupby("stock_id"):
        rows.append({
            "stock_id": stock_id,
            "appearances": int(len(group)),
            "top_1_count": int((group["rank"] <= 1).sum()),
            "top_3_count": int((group["rank"] <= 3).sum()),
            "top_5_count": int((group["rank"] <= 5).sum()),
            "top_10_count": int((group["rank"] <= 10).sum()),
            "avg_rank": float(group["rank"].mean()),
            "longest_consecutive_top10": _longest_consecutive_dates(group["signal_date"]),
            "first_signal_date": group["signal_date"].min(),
            "last_signal_date": group["signal_date"].max(),
        })

    return pd.DataFrame(rows).sort_values(
        ["top_5_count", "top_10_count", "avg_rank", "stock_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
