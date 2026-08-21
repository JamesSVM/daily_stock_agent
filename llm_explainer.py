from __future__ import annotations

"""Explain deterministic V1.5 signals with a local Ollama model.

The strategy engine remains the source of truth. This module only converts
already-computed signal facts into human-readable explanations. It never
creates, removes, or changes a trading signal.
"""

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = """You are the explanation layer for a Taiwan stock short-term trading system.

Your job is to explain deterministic signals that were already produced by the strategy engine.
You MUST NOT create a new BUY/SELL/HOLD decision, change the supplied action, invent missing facts,
or use outside market information.

Explain only:
1. why the strategy selected the stock,
2. the strongest supporting factors from the supplied numbers,
3. the main risks or weaknesses visible in those numbers,
4. a concise neutral summary.

The strategy engine is authoritative. Treat all supplied numeric fields as facts.
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
    """Build a deterministic Ollama chat request from strategy facts."""
    facts = {
        "date": signal.get("date"),
        "stock_id": signal.get("stock_id"),
        "market_regime": signal.get("market_regime"),
        "close": signal.get("close"),
        "rs20": signal.get("rs20"),
        "rs60": signal.get("rs60"),
        "drawdown_20d": signal.get("drawdown_20d"),
        "score": signal.get("score"),
        "trend_pass": signal.get("trend_pass"),
        "momentum_pass": signal.get("momentum_pass"),
        "pullback_pass": signal.get("pullback_pass"),
        "candidate": signal.get("candidate"),
        "selected": signal.get("selected"),
        "action": signal.get("action"),
        "reason": signal.get("reason"),
    }

    user_prompt = (
        "Explain this strategy output. Do not change its action and do not add outside facts.\n\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
    )

    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": EXPLANATION_SCHEMA,
        "options": {"temperature": 0, "seed": 101},
    }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        if exc.code == 404:
            raise RuntimeError(
                "Ollama returned HTTP 404 at "
                f"{url}. The server is reachable, but this endpoint was not found. "
                "Check the local Ollama server/API URL. "
                f"Response: {detail or '<empty>'}"
            ) from exc
        raise RuntimeError(
            f"Ollama HTTP {exc.code} at {url}. Response: {detail or '<empty>'}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc


def explain_signal(
    signal: dict[str, Any],
    config: ExplanationConfig | None = None,
) -> dict[str, Any]:
    """Return an explanation while preserving the deterministic strategy fields."""
    config = config or ExplanationConfig(
        url=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    )

    payload = _build_payload(signal, config)
    response = _post_json(config.url, payload, config.timeout_seconds)

    message = response.get("message") or {}
    content = message.get("content")
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
    if not isinstance(explanation["strengths"], list) or not all(
        isinstance(item, str) for item in explanation["strengths"]
    ):
        raise RuntimeError("Ollama explanation strengths must be a list of strings.")
    if not isinstance(explanation["risks"], list) or not all(
        isinstance(item, str) for item in explanation["risks"]
    ):
        raise RuntimeError("Ollama explanation risks must be a list of strings.")

    # Preserve the strategy decision exactly; the LLM is explanation-only.
    return {
        "date": signal.get("date"),
        "stock_id": signal.get("stock_id"),
        "action": signal.get("action"),
        "selected": bool(signal.get("selected", False)),
        "score": signal.get("score"),
        "rs20": signal.get("rs20"),
        "drawdown_20d": signal.get("drawdown_20d"),
        "market_regime": signal.get("market_regime"),
        **explanation,
        "model": config.model,
    }


if __name__ == "__main__":
    raise SystemExit(
        "llm_explainer.py is a library component; call explain_signal() from the daily pipeline."
    )
