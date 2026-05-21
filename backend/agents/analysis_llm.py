from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.analysis_context import build_prompt_payload

DEFAULT_ACTIONS = [
    "Compare to last week",
    "Show similar weeks",
    "Search historical reports",
]


def fallback_analysis_response(request: Any, facts: Dict[str, Any], reason: Optional[str] = None) -> Dict[str, Any]:
    rollups = facts["rollups"]
    summary = rollups["summary"]
    top_category = (rollups.get("by_category") or [{}])[0].get("category", "n/a")
    top_payee = (rollups.get("top_payees") or [{}])[0].get("payee", "n/a")
    comparison = facts["comparison"]["totals"]["delta"]
    fallback_note = reason or "The backend is ready, but no OpenAI key is configured yet, so this is a structured fallback response."

    content = (
        f"### {request.context.card_label or 'Selected card'}\n\n"
        f"- Window: **{facts['window']['start']}** to **{facts['window']['end']}**\n"
        f"- Income: **${summary['total_income']:.2f}**\n"
        f"- Expense: **${summary['total_expense']:.2f}**\n"
        f"- Net cash flow: **${summary['net_cashflow']:.2f}**\n\n"
        f"Top category is **{top_category}** and top payee is **{top_payee}**.\n\n"
        f"Week-over-week changes: income {comparison.get('income', 0):+.2f}, expense {comparison.get('expense', 0):+.2f}, net {comparison.get('net', 0):+.2f}.\n\n"
        f"{fallback_note}"
    )
    return {
        "content": content,
        "sources": facts["sources"],
        "actions": list(DEFAULT_ACTIONS),
        "facts": {k: v for k, v in facts.items() if k not in {"sources", "strategies"}},
        "retrieval_strategy": facts["strategies"],
    }


def _parse_model_payload(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def generate_analysis_response(request: Any, facts: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = (
        "You are a finance copilot embedded in a product dashboard. "
        "Use the provided facts to answer the user's question in Markdown. "
        "Return valid JSON only with keys: content, sources, actions. "
        "content should be concise and grounded. "
        "sources must be an array of objects with label and detail. "
        "actions must be an array of short follow-up labels. "
        "If the facts are insufficient, say what is missing and suggest a next step."
    )
    user_payload = build_prompt_payload(request, facts)

    llm = ChatOpenAI(
        model=os.getenv("FINANCE_CHAT_MODEL", "gpt-5.4-mini"),
        temperature=0,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_payload),
        ]
    )
    parsed = _parse_model_payload(response.content)
    content = str(parsed.get("content", "")).strip() or "I could not produce a structured reply."
    raw_sources = parsed.get("sources", [])
    raw_actions = parsed.get("actions", [])
    sources = [
        {"label": str(item.get("label", "Source")), "detail": str(item.get("detail", ""))}
        for item in raw_sources
        if isinstance(item, dict)
    ]
    actions = [str(item) for item in raw_actions if str(item).strip()]
    return {
        "content": content,
        "sources": sources or facts["sources"],
        "actions": actions or list(DEFAULT_ACTIONS),
        "facts": {k: v for k, v in facts.items() if k not in {"sources", "strategies"}},
        "retrieval_strategy": facts["strategies"],
    }
