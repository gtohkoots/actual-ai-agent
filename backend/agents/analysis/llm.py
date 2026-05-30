from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.analysis.presenter import render_analysis_chat_content

DEFAULT_ACTIONS = [
    "Compare to last week",
    "Review top categories",
    "Inspect flagged transactions",
]


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


def _tool_detail_line(tool_result: Dict[str, Any]) -> str:
    tool = tool_result.get("tool")
    result = tool_result.get("result", {})
    if tool == "get_portfolio_summary":
        summary = result.get("summary", {})
        return (
            f"Income ${summary.get('total_income', 0.0):.2f}, expense ${summary.get('total_expense', 0.0):.2f}, "
            f"net ${summary.get('net_cashflow', 0.0):.2f}."
        )
    if tool == "get_category_spend":
        categories = result.get("categories", [])
        if categories:
            top = categories[0]
            return f"Top category is {top.get('category_name', 'n/a')} at ${float(top.get('amount', 0.0)):.2f}."
        return "No expense categories were found for the selected window."
    if tool == "compare_periods":
        deltas = result.get("total_deltas", {})
        return (
            f"Period deltas: income {float(deltas.get('total_income', 0.0)):+.2f}, "
            f"expense {float(deltas.get('total_expense', 0.0)):+.2f}, net {float(deltas.get('net_cashflow', 0.0)):+.2f}."
        )
    if tool == "get_spending_drift":
        changes = result.get("top_category_changes", [])
        if changes:
            top = changes[0]
            return f"Largest category change is {top.get('category_name', 'n/a')} at {float(top.get('delta', 0.0)):+.2f}."
        return "No category drift was detected for the selected window."
    if tool == "get_transaction_slice":
        transactions = result.get("transactions", [])
        return f"Retrieved {len(transactions)} matching transactions for drill-down."
    if tool == "detect_spending_anomalies":
        anomalies = result.get("anomalies", [])
        return f"Detected {len(anomalies)} anomaly candidate(s)."
    if tool == "find_recurring_charges":
        recurring = result.get("recurring_charges", [])
        return f"Found {len(recurring)} recurring-charge candidate(s)."
    if tool == "get_recent_anomalies":
        items = result if isinstance(result, list) else []
        return f"Loaded {len(items)} recent anomaly record(s) from history."
    if tool == "search_reports":
        items = result if isinstance(result, list) else []
        return f"Matched {len(items)} historical report(s)."
    if tool == "search_documents":
        items = result if isinstance(result, list) else []
        return f"Matched {len(items)} finance artifact(s)."
    if tool == "search_past_weeks_by_category":
        items = result if isinstance(result, list) else []
        return f"Found {len(items)} historical week(s) for this category."
    if tool == "find_similar_spending_weeks":
        items = result if isinstance(result, list) else []
        return f"Found {len(items)} similar historical week(s)."
    if tool == "get_account_breakdown":
        accounts = result.get("accounts", [])
        return f"Built account breakdown across {len(accounts)} account(s)."
    return f"Executed {tool} successfully."


def _sources_from_execution(execution_result: Dict[str, Any]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for item in execution_result.get("tool_results", []):
        result = item.get("result", {})
        detail = _tool_detail_line(item)
        if isinstance(result, dict):
            period_start = result.get("period_start") or result.get("current_period", {}).get("start")
            period_end = result.get("period_end") or result.get("current_period", {}).get("end")
            if period_start and period_end:
                detail = f"{period_start} to {period_end} · {detail}"
        sources.append({"label": item.get("tool", "Source"), "detail": detail})
    for failure in execution_result.get("failures", []):
        sources.append({"label": f"{failure.get('tool', 'tool')} failed", "detail": str(failure.get("error", "Unknown error"))})
    return sources


def _fallback_heading(request: Any, analysis_request: Dict[str, Any]) -> str:
    scope = analysis_request.get("scope", "portfolio")
    if scope == "account":
        return getattr(request.context, "card_label", None) or getattr(request.context, "account_name", None) or "Selected account"
    if scope == "category":
        return analysis_request.get("entity", {}).get("category") or getattr(request.context, "selected_category", None) or "Selected category"
    if scope == "payee":
        return analysis_request.get("entity", {}).get("payee") or getattr(request.context, "selected_payee", None) or "Selected payee"
    return "Overall finances"


def _facts_payload(tool_plan: Dict[str, Any], execution_result: Dict[str, Any], analysis_request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "analysis_request": dict(analysis_request),
        "tool_plan": dict(tool_plan),
        "tool_results": list(execution_result.get("tool_results", [])),
        "failures": list(execution_result.get("failures", [])),
    }


def _fallback_structured_response(
    request: Any,
    tool_plan: Dict[str, Any],
    execution_result: Dict[str, Any],
    analysis_request: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    heading = _fallback_heading(request, analysis_request)
    findings = [_tool_detail_line(item) for item in execution_result.get("tool_results", [])[:4]]
    failures = execution_result.get("failures", [])
    risks = [f"{item.get('tool', 'tool')} failed: {item.get('error', 'Unknown error')}" for item in failures[:3]]
    opportunities = []
    if "get_category_spend" not in execution_result.get("used_tools", []):
        opportunities.append("Review category mix next to see where spending concentrates.")
    if "compare_periods" not in execution_result.get("used_tools", []):
        opportunities.append("Compare this window to the previous one to quantify change.")
    return {
        "analysis_request": dict(analysis_request),
        "summary": f"{heading}: here's the grounded analysis from the executed tools.",
        "findings": findings or ["No analysis tools returned results for this request."],
        "risks": risks,
        "opportunities": opportunities,
        "actions": list(DEFAULT_ACTIONS),
        "sources": _sources_from_execution(execution_result),
        "facts": _facts_payload(tool_plan, execution_result, analysis_request),
        "retrieval_strategy": list(execution_result.get("used_tools", [])),
        "fallback_note": reason,
    }


def _build_tool_result_prompt_payload(request: Any, tool_plan: Dict[str, Any], execution_result: Dict[str, Any], analysis_request: Dict[str, Any]) -> str:
    payload = {
        "user_message": getattr(request, "message", ""),
        "context": request.context.model_dump() if hasattr(request, "context") else {},
        "analysis_request": analysis_request,
        "tool_plan": tool_plan,
        "execution_result": execution_result,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _finalize_turn_result(base: Dict[str, Any]) -> Dict[str, Any]:
    turn_result = dict(base)
    turn_result["content"] = render_analysis_chat_content(turn_result)
    return turn_result


def fallback_analysis_response(
    request: Any,
    tool_plan: Dict[str, Any],
    execution_result: Dict[str, Any],
    analysis_request: Dict[str, Any] | None = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_request = dict(analysis_request or tool_plan.get("analysis_request", {}))
    fallback_note = reason or "The analysis runtime used the deterministic fallback summary."
    return _finalize_turn_result(
        _fallback_structured_response(
            request,
            tool_plan,
            execution_result,
            normalized_request,
            fallback_note,
        )
    )


def generate_analysis_response(
    request: Any,
    tool_plan: Dict[str, Any],
    execution_result: Dict[str, Any],
    analysis_request: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_request = dict(analysis_request or tool_plan.get("analysis_request", {}))
    system_prompt = (
        "You are a finance copilot generating an answer only from executed analysis tool results. "
        "Return valid JSON only with keys: summary, findings, risks, opportunities, actions, sources. "
        "summary should be a concise plain sentence. "
        "findings, risks, opportunities, and actions must be arrays of short strings. "
        "sources must be an array of objects with label and detail. "
        "Keep every claim grounded in the execution results. "
        "If the executed tools are insufficient, say what is missing in findings or opportunities."
    )
    user_payload = _build_tool_result_prompt_payload(request, tool_plan, execution_result, normalized_request)

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
    raw_sources = parsed.get("sources", [])
    sources = [
        {"label": str(item.get("label", "Source")), "detail": str(item.get("detail", ""))}
        for item in raw_sources
        if isinstance(item, dict)
    ]
    turn_result = {
        "analysis_request": normalized_request,
        "summary": str(parsed.get("summary", "")).strip() or "I could not produce a structured analysis summary.",
        "findings": [str(item).strip() for item in parsed.get("findings", []) if str(item).strip()],
        "risks": [str(item).strip() for item in parsed.get("risks", []) if str(item).strip()],
        "opportunities": [str(item).strip() for item in parsed.get("opportunities", []) if str(item).strip()],
        "actions": [str(item).strip() for item in parsed.get("actions", []) if str(item).strip()] or list(DEFAULT_ACTIONS),
        "sources": sources or _sources_from_execution(execution_result),
        "facts": _facts_payload(tool_plan, execution_result, normalized_request),
        "retrieval_strategy": list(execution_result.get("used_tools", [])),
    }
    return _finalize_turn_result(turn_result)
