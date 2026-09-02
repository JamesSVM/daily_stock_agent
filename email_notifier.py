from __future__ import annotations

"""Render and send the V1.6 portfolio-aware daily report by SMTP.

Notification-only: this module never places or modifies trading orders.
"""

import os
import smtplib
from email.message import EmailMessage
from typing import Any


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def render_report(
    signals: list[dict[str, Any]],
    explanations: list[dict[str, Any]],
    *,
    market_regime: str,
    signal_date: str,
) -> str:
    """Render deterministic portfolio-aware facts plus optional LLM explanations."""
    explanation_by_stock = {str(item.get("stock_id")): item for item in explanations}
    selected = [item for item in signals if item.get("selected")]
    candidates = [item for item in signals if item.get("candidate")]

    snapshot = signals[0] if signals else {}
    configured = bool(snapshot.get("configured", False))
    total_capital = snapshot.get("portfolio_total_capital")
    cash_available = snapshot.get("portfolio_cash_available")
    position_count = snapshot.get("portfolio_position_count")
    max_positions = snapshot.get("portfolio_max_positions")
    monthly_used = snapshot.get("monthly_trades_used")
    monthly_limit = snapshot.get("portfolio_max_monthly_trades")

    lines = [
        "Daily Stock Agent V1.6",
        f"Date: {signal_date}",
        f"Market regime: {market_regime}",
        "",
        "Portfolio snapshot",
        f"  Configured: {'YES' if configured else 'NO'}",
        f"  Capital: {_money(total_capital)}",
        f"  Cash available: {_money(cash_available)}",
        f"  Positions: {position_count if position_count is not None else 'N/A'} / {max_positions if max_positions is not None else 'N/A'}",
        f"  Monthly trades: {monthly_used if monthly_used is not None else 'N/A'} / {monthly_limit if monthly_limit is not None else 'N/A'}",
        "",
        "Today's decision",
    ]

    if selected:
        for index, signal in enumerate(selected, start=1):
            lines.extend(
                [
                    "",
                    f"{index}. {signal.get('stock_id')} | {signal.get('action')} | score={signal.get('score')} | TradeScore={signal.get('trade_score')}",
                    f"   Allocation: {_money(signal.get('allocation_amount'))} ({float(signal.get('allocation_pct', 0.0)) * 100:.1f}%)",
                    f"   RS20: {signal.get('rs20')} | Pullback: {signal.get('drawdown_20d')}",
                    f"   Decision: {signal.get('portfolio_reason')}",
                ]
            )
            explanation = explanation_by_stock.get(str(signal.get("stock_id")), {})
            lines.append(f"   Summary: {explanation.get('summary', 'No explanation available.')}")
            if explanation.get("strengths"):
                lines.append("   Strengths: " + "; ".join(explanation["strengths"]))
            if explanation.get("risks"):
                lines.append("   Risks: " + "; ".join(explanation["risks"]))
    else:
        lines.append("NO NEW BUY TODAY.")
        if not configured:
            lines.append("Reason: portfolio state is not configured; configure data/portfolio_state.json before enabling BUY decisions.")
        elif market_regime.upper() == "BEAR":
            lines.append("Reason: new entries are blocked in BEAR regime by default.")
        elif not candidates:
            lines.append("Reason: no stock met the V1.5 candidate rules.")
        else:
            reasons = [str(item.get("portfolio_reason")) for item in candidates if item.get("portfolio_reason")]
            if reasons:
                most_common = max(set(reasons), key=reasons.count)
                lines.append(f"Reason: portfolio constraints ({most_common}).")

    lines.extend(["", f"Trade candidates today: {len(candidates)} (candidate set capped at Top 10)"])

    watchlist = [item for item in candidates if not item.get("selected")][:5]
    if watchlist:
        lines.append("",)
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
