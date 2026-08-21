from __future__ import annotations

"""Explain deterministic V1.5 signals with a local Ollama model."""

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = """You are the explanation layer for a Taiwan stock short-term trading system.

Explain deterministic signals already produced by the strategy engine.
You MUST NOT create or change a BUY/SELL/HOLD decision, invent facts, or use outside market information.
Explain only why the strategy selected the stock, supporting factors, visible risks, and a concise neutral summary.
The strategy engine is authoritative; all supplied numeric fields are facts.
"""

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "strengths", "risks"],
}

@dataclass(frozen=True)
class ExplanationConfig:
    url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


def _build_payload(signal: dict[str, Any], config: ExplanationConfig) -> dict[str, Any]:
    facts = {key: signal.get(key) for key in (
        "date", "stock_id", "market_regime", "close", "rs20", "rs60",
        "drawdown_20d", "score", "trend_pass", "momentum_pass", "pullback_pass",
        "candidate", "selected", "action", "reason")}
    user_prompt = "Explain this strategy output. Do not change its action and do not add outside facts.\n\n" + json.dumps(
        facts, ensure_ascii=False, sort_keys=True, default=str
    )
    return {
        "model": config.model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        "stream": False,
        "format": EXPLANATION_SCHEMA,
        "options": {"temperature": 0, "seed": 101},
    }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        raise RuntimeError(f"Ollama HTTP {exc.code} at {url}. Response: {detail or '<empty>'}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc


def explain_signal(signal: dict[str, Any], config: ExplanationConfig | None = None) -> dict[str, Any]:
    config = config or ExplanationConfig(
        url=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    )
    response = _post_json(config.url, _build_payload(signal, config), config.timeout_seconds)
    content = (response.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama response did not contain message.content.")
    try:
        explanation = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama explanation was not valid JSON.") from exc
    if not isinstance(explanation, dict):
        raise RuntimeError("Ollama explanation must be a JSON object.")
    for key in ("summary", "strengths", "risks"):
        if key not in explanation:
            raise RuntimeError(f"Ollama explanation missing required field: {key}")
    if not isinstance(explanation["summary"], str):
        raise RuntimeError("Ollama explanation summary must be a string.")
    if not isinstance(explanation["strengths"], list) or not all(isinstance(x, str) for x in explanation["strengths"]):
        raise RuntimeError("Ollama explanation strengths must be a list of strings.")
    if not isinstance(explanation["risks"], list) or not all(isinstance(x, str) for x in explanation["risks"]):
        raise RuntimeError("Ollama explanation risks must be a list of strings.")
    return {
        "date": signal.get("date"), "stock_id": signal.get("stock_id"),
        "action": signal.get("action"), "selected": bool(signal.get("selected", False)),
        "score": signal.get("score"), "rs20": signal.get("rs20"),
        "drawdown_20d": signal.get("drawdown_20d"), "market_regime": signal.get("market_regime"),
        **explanation, "model": config.model,
    }

if __name__ == "__main__":
    raise SystemExit("llm_explainer.py is a library component; call explain_signal() from the daily pipeline.")
