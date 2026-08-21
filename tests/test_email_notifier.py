from email_notifier import render_report


def test_render_report_uses_signal_action_and_explanation():
    signals = [
        {
            "stock_id": "2330",
            "selected": True,
            "action": "BUY_NEXT_OPEN",
            "score": 87.0,
            "rs20": 0.18,
            "drawdown_20d": -0.04,
        }
    ]
    explanations = [
        {
            "stock_id": "2330",
            "summary": "Strong relative strength with a valid pullback.",
            "strengths": ["RS20 is strong"],
            "risks": ["Pullback can continue"],
        }
    ]

    report = render_report(
        signals,
        explanations,
        market_regime="bull",
        signal_date="2026-08-21",
    )

    assert "2330" in report
    assert "BUY_NEXT_OPEN" in report
    assert "Strong relative strength" in report
    assert "Pullback can continue" in report
    assert "LLM output is explanation-only" in report


def test_render_report_handles_no_selection():
    report = render_report(
        [],
        [],
        market_regime="neutral",
        signal_date="2026-08-21",
    )

    assert "No selected signals today." in report
