from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.documents import (
    find_similar_spending_weeks,
    get_recent_anomalies,
    search_documents,
    search_past_weeks_by_category,
    search_reports,
)
from backend.services.insights import compare_week_over_week, get_week_rollups
from backend.utils.db import get_transactions_in_date_range


class ChatMessageInput(BaseModel):
    role: str = Field(..., description="system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatContext(BaseModel):
    selected_tab: Optional[str] = Field(None, description="Selected UI tab")
    account_pid: Optional[str] = Field(None, description="Actual account pid")
    account_name: Optional[str] = Field(None, description="Actual account name")
    card_label: Optional[str] = Field(None, description="Displayed card label")
    start_date: Optional[str] = Field(None, description="Window start YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="Window end YYYY-MM-DD")
    focus_category: Optional[str] = Field(None, description="Optional focus category")
    focus_payee: Optional[str] = Field(None, description="Optional focus payee")


class ChatRequest(BaseModel):
    message: str = Field(..., description="Latest user message")
    conversation_id: Optional[str] = Field(None, description="Client conversation id")
    history: List[ChatMessageInput] = Field(default_factory=list, description="Recent conversation turns")
    context: ChatContext = Field(default_factory=ChatContext)


class ChatSource(BaseModel):
    label: str
    detail: str


class ChatResponse(BaseModel):
    conversation_id: str
    content: str
    sources: List[ChatSource] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    facts: Dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: List[str] = Field(default_factory=list)


def _current_week_range() -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


def _previous_week_range(start_date: str, end_date: str) -> tuple[str, str]:
    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    return (s - timedelta(days=7)).isoformat(), (e - timedelta(days=7)).isoformat()


def _resolve_window(context: ChatContext) -> tuple[str, str]:
    if context.start_date and context.end_date:
        return context.start_date, context.end_date
    return _current_week_range()


def _normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _split_keywords(message: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&' -]{2,}|[\w\u4e00-\u9fff]{2,}", message)
    return [token.strip() for token in tokens if token.strip()]


def _build_sources_from_weekly_rollup(rollup: Dict[str, Any]) -> List[ChatSource]:
    window = rollup.get("window", {})
    summary = rollup.get("summary", {})
    return [
        ChatSource(
            label="Current window",
            detail=f'{window.get("start")} to {window.get("end")} · income {summary.get("total_income", 0.0):.2f} · expense {summary.get("total_expense", 0.0):.2f}',
        ),
    ]


def _load_context_transactions(context: ChatContext, start_date: str, end_date: str):
    return get_transactions_in_date_range(
        start_date,
        end_date,
        join_names=True,
        dollars=True,
        account_pid=context.account_pid,
        account_name=context.account_name,
        debug=False,
    )


def _build_retrieval_pack(request: ChatRequest) -> Dict[str, Any]:
    window_start, window_end = _resolve_window(request.context)
    current_df = _load_context_transactions(request.context, window_start, window_end)
    prev_start, prev_end = _previous_week_range(window_start, window_end)
    previous_df = _load_context_transactions(request.context, prev_start, prev_end)

    rollups = get_week_rollups(
        window_start,
        window_end,
        df=current_df,
    )
    comparison = compare_week_over_week(
        window_start,
        window_end,
        df=current_df,
        previous_df=previous_df,
    )

    strategies = ["live_rollup", "week_over_week"]
    sources: List[ChatSource] = []
    sources.extend(_build_sources_from_weekly_rollup(rollups))
    sources.append(ChatSource(label="Previous week", detail=f"{prev_start} to {prev_end} comparison ready"))

    if request.context.focus_category:
        category_hits = search_past_weeks_by_category(
            request.context.focus_category,
            limit=5,
        )
        if category_hits:
            strategies.append("category_history")
            sources.append(
                ChatSource(
                    label=f'Category history: {request.context.focus_category}',
                    detail=f'{len(category_hits)} historical week(s) matched',
                )
            )
    else:
        category_hits = []

    keywords = _split_keywords(request.message)
    query_lower = request.message.lower()
    similar_weeks = []
    if any(word in query_lower for word in ["similar", "compare", "like this", "same pattern"]):
        similar_weeks = find_similar_spending_weeks(window_start, window_end, limit=3)
        if similar_weeks:
            strategies.append("similar_weeks")
            sources.append(ChatSource(label="Similar weeks", detail=f"{len(similar_weeks)} prior week(s) retrieved"))

    anomaly_hits = []
    if any(word in query_lower for word in ["anomal", "spike", "unusual", "overspend", "large"]):
        anomaly_hits = get_recent_anomalies(
            payee=request.context.focus_payee,
            category=request.context.focus_category,
            limit=5,
        )
        if anomaly_hits:
            strategies.append("anomalies")
            sources.append(ChatSource(label="Recent anomalies", detail=f"{len(anomaly_hits)} flagged transaction(s)"))

    report_hits = search_reports(request.message, limit=3)
    if report_hits:
        strategies.append("reports")
        sources.append(ChatSource(label="Historical reports", detail=f"{len(report_hits)} relevant report(s)"))

    generic_hits = search_documents(query=request.message, limit=3)
    if generic_hits:
        strategies.append("document_search")
        sources.append(ChatSource(label="Artifact search", detail=f"{len(generic_hits)} document match(es)"))

    return {
        "window": {"start": window_start, "end": window_end},
        "prev_window": {"start": prev_start, "end": prev_end},
        "rollups": rollups,
        "comparison": comparison,
        "category_hits": category_hits,
        "similar_weeks": similar_weeks,
        "anomaly_hits": anomaly_hits,
        "report_hits": report_hits,
        "generic_hits": generic_hits,
        "sources": sources,
        "strategies": strategies,
        "keywords": keywords,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {key: _json_safe(item) for key, item in value.model_dump().items()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_prompt_payload(request: ChatRequest, facts: Dict[str, Any]) -> str:
    history = request.history[-8:]
    payload = {
        "user_message": request.message,
        "history": [turn.model_dump() for turn in history],
        "context": request.context.model_dump(),
        "facts": _json_safe(facts),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fallback_response(request: ChatRequest, facts: Dict[str, Any], reason: Optional[str] = None) -> ChatResponse:
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
    actions = [
        "Compare to last week",
        "Show similar weeks",
        "Search historical reports",
    ]
    return ChatResponse(
        conversation_id=request.conversation_id or str(uuid.uuid4()),
        content=content,
        sources=facts["sources"],
        actions=actions,
        facts={k: v for k, v in facts.items() if k not in {"sources", "strategies"}},
        retrieval_strategy=facts["strategies"],
    )


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


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    facts = _build_retrieval_pack(request)
    conversation_id = request.conversation_id or str(uuid.uuid4())
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _fallback_response(request, facts)

    system_prompt = (
        "You are a finance copilot embedded in a product dashboard. "
        "Use the provided facts to answer the user's question in Markdown. "
        "Return valid JSON only with keys: content, sources, actions. "
        "content should be concise and grounded. "
        "sources must be an array of objects with label and detail. "
        "actions must be an array of short follow-up labels. "
        "If the facts are insufficient, say what is missing and suggest a next step."
    )
    user_payload = _build_prompt_payload(request, facts)

    llm = ChatOpenAI(
        model=os.getenv("FINANCE_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    try:
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
            ChatSource(label=str(item.get("label", "Source")), detail=str(item.get("detail", "")))
            for item in raw_sources
            if isinstance(item, dict)
        ]
        actions = [str(item) for item in raw_actions if str(item).strip()]
    except Exception:
        fallback = _fallback_response(
            request,
            facts,
            "The model request failed, so this is a structured fallback response.",
        )
        return fallback.model_copy(update={"conversation_id": conversation_id})

    return ChatResponse(
        conversation_id=conversation_id,
        content=content,
        sources=sources or facts["sources"],
        actions=actions or ["Compare to last week", "Show similar weeks", "Search historical reports"],
        facts={k: v for k, v in facts.items() if k not in {"sources", "strategies"}},
        retrieval_strategy=facts["strategies"],
    )
