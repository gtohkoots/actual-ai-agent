from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from backend.services.conversations import load_conversation
from backend.services.documents import (
    find_similar_spending_weeks,
    get_recent_anomalies,
    search_documents,
    search_past_weeks_by_category,
    search_reports,
)
from backend.services.insights import compare_week_over_week, get_week_rollups
from backend.utils.db import get_transactions_in_date_range


def _current_week_range() -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


def _previous_week_range(start_date: str, end_date: str) -> tuple[str, str]:
    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    return (s - timedelta(days=7)).isoformat(), (e - timedelta(days=7)).isoformat()


def _resolve_window(context: Any) -> tuple[str, str]:
    if context.start_date and context.end_date:
        return context.start_date, context.end_date
    return _current_week_range()


def _split_keywords(message: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&' -]{2,}|[\w\u4e00-\u9fff]{2,}", message)
    return [token.strip() for token in tokens if token.strip()]


def _build_sources_from_weekly_rollup(rollup: Dict[str, Any]) -> List[Dict[str, str]]:
    window = rollup.get("window", {})
    summary = rollup.get("summary", {})
    return [
        {
            "label": "Current window",
            "detail": (
                f'{window.get("start")} to {window.get("end")} '
                f'· income {summary.get("total_income", 0.0):.2f} '
                f'· expense {summary.get("total_expense", 0.0):.2f}'
            ),
        }
    ]


def _load_context_transactions(context: Any, start_date: str, end_date: str):
    return get_transactions_in_date_range(
        start_date,
        end_date,
        join_names=True,
        dollars=True,
        account_pid=context.account_pid,
        account_name=context.account_name,
        debug=False,
    )


def build_analysis_context(request: Any) -> Dict[str, Any]:
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
    sources: List[Dict[str, str]] = []
    sources.extend(_build_sources_from_weekly_rollup(rollups))
    sources.append({"label": "Previous week", "detail": f"{prev_start} to {prev_end} comparison ready"})

    if request.context.focus_category:
        category_hits = search_past_weeks_by_category(
            request.context.focus_category,
            limit=5,
        )
        if category_hits:
            strategies.append("category_history")
            sources.append(
                {
                    "label": f"Category history: {request.context.focus_category}",
                    "detail": f"{len(category_hits)} historical week(s) matched",
                }
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
            sources.append({"label": "Similar weeks", "detail": f"{len(similar_weeks)} prior week(s) retrieved"})

    anomaly_hits = []
    if any(word in query_lower for word in ["anomal", "spike", "unusual", "overspend", "large"]):
        anomaly_hits = get_recent_anomalies(
            payee=request.context.focus_payee,
            category=request.context.focus_category,
            limit=5,
        )
        if anomaly_hits:
            strategies.append("anomalies")
            sources.append({"label": "Recent anomalies", "detail": f"{len(anomaly_hits)} flagged transaction(s)"})

    report_hits = search_reports(request.message, limit=3)
    if report_hits:
        strategies.append("reports")
        sources.append({"label": "Historical reports", "detail": f"{len(report_hits)} relevant report(s)"})

    generic_hits = search_documents(query=request.message, limit=3)
    if generic_hits:
        strategies.append("document_search")
        sources.append({"label": "Artifact search", "detail": f"{len(generic_hits)} document match(es)"})

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


def conversation_history_for_prompt(request: Any) -> List[Dict[str, str]]:
    if request.history:
        return [turn.model_dump() for turn in request.history[-8:]]
    if not request.conversation_id:
        return []
    try:
        thread = load_conversation(request.conversation_id)
    except KeyError:
        return []
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in thread.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    ]
    return messages[-8:]


def build_prompt_payload(request: Any, facts: Dict[str, Any]) -> str:
    payload = {
        "user_message": request.message,
        "history": conversation_history_for_prompt(request),
        "context": request.context.model_dump(),
        "facts": _json_safe(facts),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
