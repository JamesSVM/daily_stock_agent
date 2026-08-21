from __future__ import annotations

"""End-to-end daily report pipeline.

Signal generation is deterministic. Ollama only explains selected signals.
Email is notification-only. No trading order is placed by this module.
"""

import argparse
import sqlite3
from pathlib import Path

from daily_signal import DB_PATH, DEFAULT_OUTPUT, build_signal_sheet, load_stock_data
from email_notifier import render_report, send_email
from llm_explainer import explain_signal


def run(
    db_path: str = DB_PATH,
    output_path: str = DEFAULT_OUTPUT,
    send_notification: bool = False,
) -> str:
    with sqlite3.connect(db_path) as conn:
        stock_data = load_stock_data(conn)

    signals, regime = build_signal_sheet(stock_data)
    if signals.empty:
        signal_date = "N/A"
        signal_rows: list[dict] = []
    else:
        signal_date = str(signals["date"].iloc[0])
        signal_rows = signals.to_dict(orient="records")

    selected = [row for row in signal_rows if row.get("selected")]
    explanations = [explain_signal(row) for row in selected]
    report = render_report(
        signal_rows,
        explanations,
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
    if send_notification:
        print("Email: sent")
    else:
        print("Email: not sent (use --send-email to enable)")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V1.5 daily report pipeline")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Signal output path; report uses .txt")
    parser.add_argument("--send-email", action="store_true", help="Send the report by SMTP")
    args = parser.parse_args()
    run(db_path=args.db, output_path=args.output, send_notification=args.send_email)


if __name__ == "__main__":
    main()
