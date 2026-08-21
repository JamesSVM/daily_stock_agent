from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from llm_explainer import ExplanationConfig, _build_payload, explain_signal


SIGNAL = {
    "date": "2026-08-21",
    "stock_id": "2330",
    "market_regime": "bull",
    "close": 100.0,
    "rs20": 0.15,
    "rs60": 0.21,
    "drawdown_20d": -0.05,
    "score": 82.0,
    "trend_pass": True,
    "momentum_pass": True,
    "pullback_pass": True,
    "candidate": True,
    "selected": True,
    "action": "BUY_NEXT_OPEN",
    "reason": "candidate",
}


def test_payload_contains_strategy_facts_and_explanation_only_instruction() -> None:
    config = ExplanationConfig(model="test-model")
    payload = _build_payload(SIGNAL, config)

    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["format"]["type"] == "object"

    user_content = payload["messages"][1]["content"]
    assert "2330" in user_content
    assert "BUY_NEXT_OPEN" in user_content
    assert "Do not change its action" in user_content


def test_explainer_preserves_strategy_action() -> None:
    response = {
        "message": {
            "content": json.dumps(
                {
                    "summary": "Strong relative strength with a controlled pullback.",
                    "strengths": ["RS20 remains positive."],
                    "risks": ["Pullback could deepen."],
                },
                ensure_ascii=False,
            )
        }
    }

    with patch("llm_explainer._post_json", return_value=response):
        result = explain_signal(SIGNAL, ExplanationConfig(model="test-model"))

    assert result["action"] == SIGNAL["action"]
    assert result["selected"] is True
    assert result["score"] == SIGNAL["score"]
    assert result["summary"]
    assert result["strengths"]
    assert result["risks"]


def test_invalid_llm_json_is_rejected() -> None:
    response = {"message": {"content": "not-json"}}

    with patch("llm_explainer._post_json", return_value=response):
        with pytest.raises(RuntimeError, match="valid JSON"):
            explain_signal(SIGNAL, ExplanationConfig(model="test-model"))


def test_missing_explanation_field_is_rejected() -> None:
    response = {
        "message": {
            "content": json.dumps(
                {
                    "summary": "summary",
                    "strengths": [],
                }
            )
        }
    }

    with patch("llm_explainer._post_json", return_value=response):
        with pytest.raises(RuntimeError, match="missing required field: risks"):
            explain_signal(SIGNAL, ExplanationConfig(model="test-model"))
