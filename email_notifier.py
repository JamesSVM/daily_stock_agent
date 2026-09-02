from __future__ import annotations

"""Render and send the V1.6 portfolio-aware daily report by SMTP.

Notification-only: this module never places or modifies trading orders.
"""

import argparse
import os
import smtplib
from email.message import EmailMessage
from typing import Any


def render_report(
    signals: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    *,
    market_regime: str,
    signal_date: str,
) -> str:
    """Render the quantitative candidate summary and AI-selected Top 3."""
    candidates = [item for item in signals if item.get("candidate")]
    eligible = [item for item in signals if item.get("selected")]
    signal_by_stock = {str(item.get("stock_id")): item for item in signals}
    snapshot = signals[0] if signals else {}
    total_capital = snapshot.get("total_capital", 0)
    position_count = snapshot.get("position_count", 0)

    lines = [
        "Daily Stock Agent V1.6.1",
        f"Date: {signal_date}",
        f"Market regime: {market_regime}",
        "",
        "Portfolio context",
        f"  Total capital: ${float(total_capital):,.0f}" if total_capital else "  Total capital: not configured",
        f"  Known holdings: {position_count}",
        "  Cash availability / position sizing: not used",
        "",
        "AI TOP 3 RECOMMENDATIONS",
    ]

    if recommendations:
        for rec in recommendations:
            stock_id = str(rec.get("stock_id"))
            signal = signal_by_stock.get(stock_id, {})
            lines.extend(
                [
                    "",
                    f"{rec.get('rank')}. {stock_id} | {signal.get('action')} | score={signal.get('score')}",
                    f"   RS20: {signal.get('rs20')} | RS60: {signal.get('rs60')} | Pullback: {signal.get('drawdown_20d')}",
                    f"   Why selected: {rec.get('reason')}",
                ]
            )
            if rec.get("strengths"):
                lines.append("   Strengths: " + "; ".join(str(x) for x in rec["strengths"]))
            if rec.get("risks"):
                lines.append("   Key risk: " + "; ".join(str(x) for x in rec["risks"]))
    else:
        lines.append("NO TOP-3 RECOMMENDATION TODAY.")
        if market_regime.upper() == "BEAR":
            lines.append("Reason: new entries are blocked in BEAR regime by default.")
        elif not candidates:
            lines.append("Reason: no stock met the V1.5 candidate rules.")
        elif not eligible:
            lines.append("Reason: no candidate passed the V1.6 entry gates.")

    lines.extend(
        [
            "",
            "Quantitative candidate summary",
            f"  V1.5 candidates: {len(candidates)}",
            f"  V1.6 eligible for AI ranking: {len(eligible)}",
            f"  AI recommendations: {len(recommendations)}",
            "",
            "The quantitative engine determines eligibility. AI only prioritizes the eligible pool into Top 3 and explains the selection.",
        ]
    )
    return "\n".join(lines)


def send_email(
    body: str,
    *,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    sender: str | None = None,
    recipient: str | None = None,
    subject: str = "Daily Stock Agent Report",
) -> None:
    """Send a plain-text email using explicit arguments or environment variables."""
    host = smtp_host or os.environ["SMTP_HOST"]
    port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
    user = smtp_user or os.environ["SMTP_USER"]
    password = smtp_password or os.environ["SMTP_PASSWORD"]
    sender = sender or os.environ["EMAIL_FROM"]
    recipient = recipient or os.environ["EMAIL_TO"]

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(message)


def _send_failure_email(error_text: str) -> None:
    body = (
        "Daily Stock Agent ALERT\n\n"
        "The scheduled pipeline failed before a normal daily report could be delivered.\n\n"
        f"Reason: {error_text}\n"
    )
    send_email(body, subject="Daily Stock Agent ALERT - Pipeline Failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a Daily Stock Agent failure alert")
    parser.add_argument("--failure", help="Failure reason to include in the alert")
    args = parser.parse_args()
    if args.failure:
        _send_failure_email(args.failure)
