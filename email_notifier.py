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
    explanations: list[dict[str, Any]],
    *,
    market_regime: str,
    signal_date: str,
) -> str:
    """Render deterministic signal facts plus optional LLM explanations."""
    explanation_by_stock = {str(item.get("stock_id")): item for item in explanations}
    selected = [item for item in signals if item.get("selected")]
    candidates = [item for item in signals if item.get("candidate")]
    snapshot = signals[0] if signals else {}
    total_capital = snapshot.get("total_capital", 0)
    position_count = snapshot.get("position_count", 0)

    lines = [
        "Daily Stock Agent V1.6",
        f"Date: {signal_date}",
        f"Market regime: {market_regime}",
        "",
        "Portfolio context",
        f"  Total capital: ${float(total_capital):,.0f}" if total_capital else "  Total capital: not configured",
        f"  Known holdings: {position_count}",
        "  Cash availability / position sizing: not used in V1.6",
        "",
        "Today's decision",
    ]

    if selected:
        for index, signal in enumerate(selected, start=1):
            stock_id = str(signal.get("stock_id"))
            explanation = explanation_by_stock.get(stock_id, {})
            lines.extend(
                [
                    "",
                    f"{index}. {stock_id} | {signal.get('action')} | score={signal.get('score')} | TradeScore={signal.get('trade_score')}",
                    f"   RS20: {signal.get('rs20')} | Pullback: {signal.get('drawdown_20d')}",
                    f"   Decision: {signal.get('portfolio_reason')}",
                    f"   Summary: {explanation.get('summary', 'No explanation available.')}",
                ]
            )
            if explanation.get("strengths"):
                lines.append("   Strengths: " + "; ".join(explanation["strengths"]))
            if explanation.get("risks"):
                lines.append("   Risks: " + "; ".join(explanation["risks"]))
    else:
        lines.append("NO NEW BUY TODAY.")
        if market_regime.upper() == "BEAR":
            lines.append("Reason: new entries are blocked in BEAR regime by default.")
        elif not candidates:
            lines.append("Reason: no stock met the V1.5 candidate rules.")
        else:
            reasons = [str(item.get("portfolio_reason")) for item in candidates if item.get("portfolio_reason")]
            if reasons:
                lines.append(f"Reason: {max(set(reasons), key=reasons.count)}.")

    lines.extend(["", f"Trade candidates shown: {len(candidates)} (Top 10 display cap)"])
    watchlist = [item for item in candidates if not item.get("selected")][:5]
    if watchlist:
        lines.append("")
        lines.append("Watchlist")
        for item in watchlist:
            lines.append(
                f"  {item.get('stock_id')} | score={item.get('score')} | action={item.get('action')} | {item.get('portfolio_reason')}"
            )

    lines.extend(["", "Strategy decisions are deterministic; LLM output is explanation-only."])
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
