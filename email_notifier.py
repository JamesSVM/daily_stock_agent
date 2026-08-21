from __future__ import annotations

"""Render and send the daily V1.5 report by SMTP.

This module is notification-only. It does not create or modify trading signals.
"""

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
    """Render a deterministic plain-text report from signal facts and explanations."""
    explanation_by_stock = {str(item.get("stock_id")): item for item in explanations}
    lines = [
        "Daily Stock Agent",
        f"Date: {signal_date}",
        f"Market regime: {market_regime}",
        "",
        "Selected signals",
    ]

    selected = [item for item in signals if item.get("selected")]
    if not selected:
        lines.append("No selected signals today.")
    else:
        for index, signal in enumerate(selected, start=1):
            stock_id = str(signal.get("stock_id"))
            explanation = explanation_by_stock.get(stock_id, {})
            lines.extend(
                [
                    "",
                    f"{index}. {stock_id} | action={signal.get('action')} | score={signal.get('score')} | RS20={signal.get('rs20')}",
                    f"   Pullback: {signal.get('drawdown_20d')}",
                    f"   Summary: {explanation.get('summary', 'No explanation available.')}",
                ]
            )
            strengths = explanation.get("strengths", [])
            risks = explanation.get("risks", [])
            if strengths:
                lines.append("   Strengths: " + "; ".join(strengths))
            if risks:
                lines.append("   Risks: " + "; ".join(risks))

    lines.extend(["", "Strategy decision is deterministic; LLM output is explanation-only."])
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
    """Send the rendered report using SMTP credentials from arguments or environment."""
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
