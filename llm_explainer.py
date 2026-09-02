from __future__ import annotations

"""Explain deterministic signals and rank eligible candidates with local Ollama."""

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_SECONDS = 120
TOP_RECOMMENDATIONS = 3

SYSTEM_PROMPT = """你是台灣股票短線交易系統的「AI 優先排序與解釋層」。

策略引擎已經先決定哪些股票通過量化進場條件。你的唯一工作，是從「已通過量化條件的候選池」中，選出最值得優先考慮的 3 檔股票，並解釋為什麼。

嚴格規則：
1. 只能從提供的候選股票中選擇，不得自行加入股票。
2. 必須選出最多 3 檔；候選不足 3 檔時選全部候選。
3. 不得修改任何策略引擎的 action，也不得創造新的 BUY/SELL/HOLD/WATCH/NO_TRADE 決策。
4. 不得使用外部市場資訊、新聞、基本面資料或你自己的即時知識。
5. 只能根據提供的量化欄位排序，例如 score、RS20、RS60、drawdown_20d、trend_pass、momentum_pass、pullback_pass 與 market_regime。
6. 對每一檔入選股票說明最重要的支持因素，以及一個主要風險或限制。
7. 使用繁體中文（zh-TW）。股票代號與程式欄位名稱保持原樣。
8. 不要聲稱未提供的未來報酬率、勝率或其他統計結果。

你不是交易執行器，也不是預測未來的模型；你是在既有量化候選池中做「優先級排序」。"""

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "maxItems": TOP_RECOMMENDATIONS,
            "items": {
                "type": "object",
                "properties": {
                    "stock_id": {"type": "string"},
                    "rank": {"type": "integer", "minimum": 1, "maximum": TOP_RECOMMENDATIONS},
                    "reason": {"type": "string"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["stock_id", "rank", "reason", "strengths", "risks"],
            },
        }
    },
    "required": ["recommendations"],
}

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
        raise RuntimeError(
            f"Ollama HTTP {exc.code} at {url}. Response: {detail or '<empty>'}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc


def _config_from_env() -> ExplanationConfig:
    return ExplanationConfig(
        url=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    )


def _candidate_facts(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        key: signal.get(key)
        for key in (
            "date", "stock_id", "market_regime", "close", "rs20", "rs60",
            "drawdown_20d", "score", "trade_score", "trend_pass", "momentum_pass",
            "pullback_pass", "candidate", "selected", "action", "reason",
            "portfolio_reason",
        )
    }


def rank_top_candidates(
    candidates: list[dict[str, Any]],
    config: ExplanationConfig | None = None,
) -> list[dict[str, Any]]:
    """Ask Ollama to rank the eligible candidate pool and return at most Top 3."""
    if not candidates:
        return []
    config = config or _config_from_env()
    facts = [_candidate_facts(candidate) for candidate in candidates]
    user_prompt = (
        "從以下已通過量化進場條件的候選池中，選出最值得優先考慮的 Top 3。\n"
        "只能使用提供的資料排序，不得加入外部資訊。\n\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
    )
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": RANKING_SCHEMA,
        "options": {"temperature": 0, "seed": 101},
    }
    response = _post_json(config.url, payload, config.timeout_seconds)
    content = (response.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama ranking response did not contain message.content.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama ranking response was not valid JSON.") from exc
    recommendations = result.get("recommendations") if isinstance(result, dict) else None
    if not isinstance(recommendations, list):
        raise RuntimeError("Ollama ranking response missing recommendations list.")

    candidate_ids = {str(item.get("stock_id")) for item in candidates}
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        stock_id = str(item.get("stock_id", ""))
        if not stock_id or stock_id not in candidate_ids or stock_id in seen:
            continue
        seen.add(stock_id)
        validated.append(
            {
                "stock_id": stock_id,
                "rank": len(validated) + 1,
                "reason": str(item.get("reason") or "AI prioritized this candidate based on the supplied quantitative signals."),
                "strengths": item.get("strengths") if isinstance(item.get("strengths"), list) else [],
                "risks": item.get("risks") if isinstance(item.get("risks"), list) else [],
                "model": config.model,
            }
        )
        if len(validated) >= TOP_RECOMMENDATIONS:
            break

    if not validated:
        raise RuntimeError("Ollama ranking returned no valid candidate stock IDs.")
    return validated


def explain_signal(signal: dict[str, Any], config: ExplanationConfig | None = None) -> dict[str, Any]:
    """Explain one already-ranked recommendation with local Ollama."""
    config = config or _config_from_env()
    facts = _candidate_facts(signal)
    user_prompt = (
        "請用繁體中文（zh-TW）解釋這一檔已被 AI 排入 Top 3 的候選股票。\n"
        "不得改變 action，也不得加入外部資訊。\n\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
    )
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": EXPLANATION_SCHEMA,
        "options": {"temperature": 0, "seed": 101},
    }
    response = _post_json(config.url, payload, config.timeout_seconds)
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
    if not isinstance(explanation["strengths"], list) or not all(
        isinstance(x, str) for x in explanation["strengths"]
    ):
        raise RuntimeError("Ollama explanation strengths must be a list of strings.")
    if not isinstance(explanation["risks"], list) or not all(
        isinstance(x, str) for x in explanation["risks"]
    ):
        raise RuntimeError("Ollama explanation risks must be a list of strings.")
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
