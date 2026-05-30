from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.analysis.tools import get_analysis_tool_registry, list_analysis_tool_specs


DEFAULT_MODEL = os.getenv("FINANCE_ANALYSIS_PLANNER_MODEL", os.getenv("FINANCE_CHAT_MODEL", "gpt-5.4-mini"))


def _current_week_range() -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


def _resolve_window(context: Any) -> tuple[str, str]:
    if getattr(context, "start_date", None) and getattr(context, "end_date", None):
        return context.start_date, context.end_date
    return _current_week_range()


def _previous_matching_window(period_start: str, period_end: str) -> tuple[str, str]:
    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    window_days = (end - start).days + 1
    baseline_end = start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=window_days - 1)
    return baseline_start.isoformat(), baseline_end.isoformat()


def _context_account_args(context: Any, entity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "account_pid": entity.get("account_pid") or getattr(context, "selected_account_pid", None) or getattr(context, "account_pid", None),
            "account_name": entity.get("account_name") or getattr(context, "selected_account_name", None) or getattr(context, "account_name", None),
        }.items()
        if value
    }


def _transaction_account_args(account_args: Dict[str, Any]) -> Dict[str, Any]:
    if account_args.get("account_name"):
        return {"account_name": account_args["account_name"]}
    return {}


def _portfolio_summary_call(window_start: str, window_end: str, account_args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tool": "get_portfolio_summary",
        "args": {"period_start": window_start, "period_end": window_end, **account_args},
    }


def _category_spend_call(window_start: str, window_end: str, account_args: Dict[str, Any], *, limit: int = 8) -> Dict[str, Any]:
    return {
        "tool": "get_category_spend",
        "args": {"period_start": window_start, "period_end": window_end, "limit": limit, **account_args},
    }


def _comparison_call(window_start: str, window_end: str, prev_start: str, prev_end: str) -> Dict[str, Any]:
    return {
        "tool": "compare_periods",
        "args": {
            "current_start": window_start,
            "current_end": window_end,
            "previous_start": prev_start,
            "previous_end": prev_end,
        },
    }


def _transaction_slice_call(
    window_start: str,
    window_end: str,
    *,
    category_name: str | None = None,
    payee: str | None = None,
    account_args: Dict[str, Any],
    limit: int = 25,
) -> Dict[str, Any]:
    return {
        "tool": "get_transaction_slice",
        "args": {
            "period_start": window_start,
            "period_end": window_end,
            "limit": limit,
            **({"category_name": category_name} if category_name else {}),
            **({"payee": payee} if payee else {}),
            **_transaction_account_args(account_args),
        },
    }


def _category_scope_plan(intent: str, category_name: str | None, window_start: str, window_end: str, prev_start: str, prev_end: str, account_args: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    if intent == "comparison":
        tool_calls.append(_comparison_call(window_start, window_end, prev_start, prev_end))
    elif intent == "anomaly_review":
        tool_calls.append({"tool": "detect_spending_anomalies", "args": {"period_start": window_start, "period_end": window_end}})
        if category_name:
            tool_calls.append({"tool": "get_recent_anomalies", "args": {"category": category_name, "limit": 5}})
    elif intent == "cashflow_review":
        tool_calls.append(_portfolio_summary_call(window_start, window_end, account_args))

    tool_calls.append(_category_spend_call(window_start, window_end, account_args))
    tool_calls.append(
        _transaction_slice_call(
            window_start,
            window_end,
            category_name=category_name,
            account_args=account_args,
            limit=25,
        )
    )
    if category_name:
        tool_calls.append({"tool": "search_past_weeks_by_category", "args": {"category": category_name, "limit": 5}})
    return tool_calls


def _payee_scope_plan(intent: str, payee: str | None, window_start: str, window_end: str, prev_start: str, prev_end: str, account_args: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    if intent == "comparison":
        tool_calls.append(_comparison_call(window_start, window_end, prev_start, prev_end))
    elif intent == "anomaly_review":
        tool_calls.append({"tool": "detect_spending_anomalies", "args": {"period_start": window_start, "period_end": window_end}})
        if payee:
            tool_calls.append({"tool": "get_recent_anomalies", "args": {"payee": payee, "limit": 5}})
    elif intent == "cashflow_review":
        tool_calls.append(_portfolio_summary_call(window_start, window_end, account_args))
    elif intent == "trend_review" and payee:
        tool_calls.append({"tool": "search_reports", "args": {"query": payee, "start_date": window_start, "end_date": window_end, "limit": 5}})

    tool_calls.append(
        _transaction_slice_call(
            window_start,
            window_end,
            payee=payee,
            account_args=account_args,
            limit=25,
        )
    )
    return tool_calls


def _fallback_tool_plan(request: Any, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
    context = getattr(request, "context", None)
    scope = analysis_request.get("scope", "portfolio")
    intent = analysis_request.get("intent", "summary")
    entity = dict(analysis_request.get("entity", {}))
    window_start, window_end = _resolve_window(context)
    prev_start, prev_end = _previous_matching_window(window_start, window_end)
    account_args = _context_account_args(context, entity)

    tool_calls: List[Dict[str, Any]] = []

    if scope == "payee":
        tool_calls = _payee_scope_plan(intent, entity.get("payee") or getattr(context, "selected_payee", None), window_start, window_end, prev_start, prev_end, account_args)
    elif scope == "category":
        tool_calls = _category_scope_plan(intent, entity.get("category") or getattr(context, "selected_category", None), window_start, window_end, prev_start, prev_end, account_args)
    elif intent == "summary":
        tool_calls = [
            _portfolio_summary_call(window_start, window_end, account_args if scope == "account" else {}),
            _category_spend_call(window_start, window_end, account_args if scope == "account" else {}),
        ]
    elif intent == "comparison":
        tool_calls = [
            _comparison_call(window_start, window_end, prev_start, prev_end),
            _category_spend_call(window_start, window_end, account_args if scope == "account" else {}),
        ]
    elif intent == "trend_review":
        tool_calls = [
            {"tool": "get_spending_drift", "args": {"period_start": window_start, "period_end": window_end}},
            _category_spend_call(window_start, window_end, account_args if scope == "account" else {}),
        ]
    elif intent == "anomaly_review":
        tool_calls = [
            {"tool": "detect_spending_anomalies", "args": {"period_start": window_start, "period_end": window_end}},
            {"tool": "get_recent_anomalies", "args": {"limit": 5}},
        ]
    elif intent == "cashflow_review":
        tool_calls = [
            _portfolio_summary_call(window_start, window_end, account_args if scope == "account" else {}),
            {"tool": "get_account_breakdown", "args": {"period_start": window_start, "period_end": window_end}},
        ]
    elif intent == "category_deep_dive":
        category_name = entity.get("category") or getattr(context, "selected_category", None)
        tool_calls = _category_scope_plan(intent, category_name, window_start, window_end, prev_start, prev_end, account_args)
    else:
        tool_calls = [_portfolio_summary_call(window_start, window_end, account_args if scope == "account" else {})]

    return {
        "scope": scope,
        "intent": intent,
        "tool_calls": tool_calls,
        "reasoning": "Deterministic fallback analysis tool plan based on interpreted scope and intent.",
        "planning_mode": "fallback",
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


def _normalize_tool_plan(request: Any, analysis_request: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    registry = get_analysis_tool_registry()
    fallback = _fallback_tool_plan(request, analysis_request)
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return fallback

    normalized_calls: List[Dict[str, Any]] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool", "")).strip()
        if tool_name not in registry:
            continue
        spec = registry[tool_name]
        raw_args = item.get("args") if isinstance(item.get("args"), dict) else {}
        properties = spec.arg_schema.get("properties", {})
        cleaned_args = {key: value for key, value in raw_args.items() if key in properties and value is not None}
        required = spec.arg_schema.get("required", [])
        if any(key not in cleaned_args for key in required):
            continue
        normalized_calls.append({"tool": tool_name, "args": cleaned_args})

    if not normalized_calls:
        return fallback

    return {
        "scope": str(payload.get("scope") or analysis_request.get("scope", "portfolio")),
        "intent": str(payload.get("intent") or analysis_request.get("intent", "summary")),
        "tool_calls": normalized_calls,
        "reasoning": str(payload.get("reasoning") or "LLM-generated analysis tool plan."),
        "planning_mode": "model",
    }


def plan_analysis_tool_calls(request: Any, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_tool_plan(request, analysis_request)

    system_prompt = (
        "You are planning deterministic finance-analysis tool calls. "
        "Given the user request, interpreted scope/intent, UI context, and available tools, return valid JSON only. "
        "Return keys: scope, intent, reasoning, tool_calls. "
        "tool_calls must be an array of {tool, args}. "
        "Only choose tools from the provided registry. "
        "Do not invent tool names or arguments. "
        "Prefer the smallest useful tool set that can answer the question accurately."
    )
    payload = {
        "user_message": getattr(request, "message", ""),
        "analysis_request": analysis_request,
        "context": request.context.model_dump() if hasattr(request, "context") else {},
        "available_tools": list_analysis_tool_specs(),
    }

    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]
    )
    parsed = _parse_model_payload(response.content)
    return _normalize_tool_plan(request, analysis_request, parsed)
