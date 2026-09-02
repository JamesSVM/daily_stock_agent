from __future__ import annotations

"""End-to-end V1.6 daily report pipeline.

Quantitative signal generation and portfolio eligibility are deterministic.
Local Ollama only prioritizes the eligible candidate pool into Top 3 and
explains those three recommendations. Email is notification-only.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from daily_signal import (
    DB_PATH,
    DEFAULT_OUTPUT,
    DEFAULT_PORTFOLIO_STATE,
    build_signal_sheet,
    load_stock_data,
)
from email_notifier import render_report, send_email
from llm_explainer import rank_top_candidates

TOP_RECOMMENDATIONS = 3


def _fallback_top3(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic fallback when Ollama is unavailable."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("rs20") or 0.0),
            float(item.get("rs60") or 0.0),
        ),
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked[:TOP_RECOMMENDATIONS], start=1):
        results.append(
            {
                "stock_id": str(item.get("stock_id")),
                "rank": rank,
                "reason": "Ollama unavailable; deterministic fallback prioritized higher Score, then RS20 and RS60.",
                "strengths": [
                    f"Score={item.get('score')}",
                    f"RS20={item.get('rs20')}",
                    f"RS60={item.get('rs60')}",
                ],
                "risks": ["AI ranking unavailable for this report."],
                "model": "fallback",
            }
        )
    return results


def run(
    db_path: str = DB_PATH,
    output_path: str = DEFAULT_OUTPUT,
    portfolio_state_path: str = DEFAULT_PORTFOLIO_STATE,
    send_notification: bool = False,
) -> str:
    with sqlite3.connect(db_path) as conn:
        stock_data = load_stock_data(conn)

    signals, regime = build_signal_sheet(
        stock_data,
        portfolio_state_path=portfolio_state_path,
    )
    if signals.empty:
        signal_date = "N/A"
        signal_rows: list[dict] = []
    else:
        signal_date = str(signals["date"].iloc[0])
        signal_rows = signals.to_dict(orient="records")

    eligible = [row for row in signal_rows if row.get("selected")]
    recommendations: list[dict[str, Any]] = []
    if eligible:
        try:
            recommendations = rank_top_candidates(eligible)
        except Exception as error:  # noqa: BLE001 - LLM is optional to report delivery
            print(
                "AI Top-3 ranking warning: "
                f"{type(error).__name__}: {error}"
            )
            recommendations = _fallback_top3(eligible)

    recommendation_by_stock = {
        str(item.get("stock_id")): item for item in recommendations
    }
    for row in signal_rows:
        rec = recommendation_by_stock.get(str(row.get("stock_id")))
        row["ai_recommended"] = bool(rec)
        row["ai_rank"] = rec.get("rank") if rec else None

    report = render_report(
        signal_rows,
        recommendations,
        market_regime=regime,
        signal_date=signal_date,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".txt").write_text(report, encoding="utf-8")

    if send_notification:
        send_email(report, subject=f"Daily Stock Agent Report - {signal_date}")

    print(report)
    print(f"Saved: {output.with_suffix('.txt')}")
    print("Email: sent" if send_notification else "Email: not sent (use --send-email to enable)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V1.6 daily portfolio-aware report pipeline")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Signal CSV output path; report uses .txt")
    parser.add_argument(
        "--portfolio-state",
        default=DEFAULT_PORTFOLIO_STATE,
        help="JSON file containing capital, positions and trade history",
    )
    parser.add_argument("--send-email", action="store_true", help="Send the report by SMTP")
    args = parser.parse_args()
    run(
        db_path=args.db,
        output_path=args.output,
        portfolio_state_path=args.portfolio_state,
        send_notification=args.send_email,
    )


if __name__ == "__main__":
    main()
