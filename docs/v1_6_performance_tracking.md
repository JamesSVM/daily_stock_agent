# V1.6 Performance Tracking

V1.6 measures whether the frozen V1.5 ranking has predictive value. It is deliberately separate from the live entry/exit engine.

## Ranking buckets

Every trading date is ranked using the same V1.5 ordering:

1. candidate stocks first
2. RS20 descending
3. score descending
4. stock ID ascending as a deterministic tie-breaker

Only ranks 1-10 are stored as signal events. The reports aggregate those events into cumulative buckets:

- Top 1 = rank 1
- Top 3 = ranks 1-3
- Top 5 = ranks 1-5
- Top 10 = ranks 1-10

## Duplicate stocks

A stock appearing on multiple dates creates multiple independent signal events. This is intentional: the question is whether each daily signal has predictive value.

Persistence is measured separately using appearance count, Top-1/3/5/10 counts, average rank, and longest consecutive Top-10 streak.

## Forward-return definition

Signal date **T** is the close at which the ranking is known. The modeled entry is **T+1 open**.

Returns are measured from that entry open to the close after 1, 3, 5 and 10 trading sessions.

No future data is used to determine the rank on T.

## Benchmark

The tracker attempts to use Yahoo Finance `^TWII` as the TAIEX benchmark. If the benchmark is unavailable, stock-level signal returns are still generated and benchmark/excess-return fields remain missing.

## Outputs

`python scripts/build_performance_tracking.py` writes:

- `reports/performance/signal_events_top10.csv`
- `reports/performance/performance_summary.csv`
- `reports/performance/persistence_summary.csv`

The daily agent runs this step after the live report. A V1.6 analytics failure is non-blocking: the existing V1.5 signal and email delivery still succeed.

## Important interpretation

Signal-event performance and portfolio performance are different. A stock that reappears on multiple days is counted multiple times in signal-event statistics, but a future portfolio simulator should prevent accidental duplicate live positions.
