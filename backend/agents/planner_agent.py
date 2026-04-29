from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from backend.agents.llm import generate_planner_response
from backend.agents.mcp_client import call_tool_payload, get_multiple_resource_payloads
from backend.agents.prompts import build_planner_prompt_context
from backend.agents.planner_state import PlannerAgentState


def run_planner_agent(user_message: str, *, db_path: Optional[str] = None) -> dict[str, Any]:
    """Run the planner agent over MCP-backed context and LLM synthesis."""
    resources = [
        "planner://budget/active-plan",
        "planner://budget/current-status",
    ]
    payloads = get_multiple_resource_payloads(resources, db_path=db_path)
    active_plan = payloads["planner://budget/active-plan"]
    current_status = payloads["planner://budget/current-status"]

    state = PlannerAgentState(
        user_message=user_message,
        active_budget_plan=active_plan,
        budget_status=current_status,
        used_resources=resources,
    )
    review_mode = _detect_review_mode(user_message)
    if review_mode == "historical_review":
        state.tool_results = _load_historical_tool_results(user_message, db_path=db_path)
        state.used_tools = list(state.tool_results.keys())

    state.prompt_context = build_planner_prompt_context(
        user_message=user_message,
        active_budget_plan=active_plan,
        budget_status=current_status,
        review_mode=review_mode,
        tool_results=state.tool_results,
    )
    state.model_response = generate_planner_response(state.prompt_context)
    state.summary = state.model_response.get("summary", "")
    state.highlights = list(state.model_response.get("highlights", []))
    state.next_action = state.model_response.get("next_action", "")

    return _serialize_state(state)


def _serialize_state(state: PlannerAgentState) -> dict[str, Any]:
    """Return a stable response shape for CLI use and future app integration."""
    return {
        "user_message": state.user_message,
        "summary": state.summary,
        "highlights": state.highlights,
        "next_action": state.next_action,
        "used_resources": state.used_resources,
        "used_tools": state.used_tools,
        "tool_results": state.tool_results,
        "prompt_context": state.prompt_context,
        "model_response": state.model_response,
        "active_budget_plan": state.active_budget_plan,
        "budget_status": state.budget_status,
    }


def _detect_review_mode(user_message: str) -> str:
    """Choose between budget review and historical spending review."""
    normalized = user_message.strip().lower()
    historical_markers = [
        "last month",
        "previous month",
        "past month",
        "review spending",
        "spending behavior",
    ]
    if any(marker in normalized for marker in historical_markers):
        return "historical_review"
    return "budget_review"


def _load_historical_tool_results(user_message: str, *, db_path: Optional[str] = None) -> dict[str, Any]:
    """Load the MCP tool payloads needed for a historical spending review."""
    period_start, period_end = _resolve_historical_window(user_message)
    baseline_start, baseline_end = _previous_matching_window(period_start, period_end)
    tool_arguments = {
        "get_portfolio_summary": {
            "period_start": period_start,
            "period_end": period_end,
        },
        "get_category_spend": {
            "period_start": period_start,
            "period_end": period_end,
        },
        "get_account_breakdown": {
            "period_start": period_start,
            "period_end": period_end,
        },
        "get_spending_drift": {
            "period_start": period_start,
            "period_end": period_end,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
        },
    }
    return {
        tool_name: call_tool_payload(tool_name, arguments=arguments, db_path=db_path)
        for tool_name, arguments in tool_arguments.items()
    }


def _resolve_historical_window(user_message: str) -> tuple[str, str]:
    """Resolve the historical analysis window from a narrow set of supported phrases."""
    normalized = user_message.strip().lower()
    if "last month" in normalized or "previous month" in normalized or "past month" in normalized:
        today = date.today()
        current_month_start = today.replace(day=1)
        previous_month_end = current_month_start - timedelta(days=1)
        previous_month_start = previous_month_end.replace(day=1)
        return previous_month_start.isoformat(), previous_month_end.isoformat()

    today = date.today()
    start = today - timedelta(days=29)
    return start.isoformat(), today.isoformat()


def _previous_matching_window(period_start: str, period_end: str) -> tuple[str, str]:
    """Return the previous window with the same inclusive duration for drift comparisons."""
    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    duration_days = (end - start).days + 1
    baseline_end = start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=duration_days - 1)
    return baseline_start.isoformat(), baseline_end.isoformat()
