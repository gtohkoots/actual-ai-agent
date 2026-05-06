from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Optional

from backend.agents.llm import generate_planner_response, interpret_planner_turn_intent
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
    state.turn_intent = interpret_planner_turn_intent(user_message, has_pending_recommendation=False)
    review_mode = state.turn_intent.get("intent", "budget_review")
    if review_mode == "historical_review":
        state.tool_results = _load_historical_tool_results(
            user_message,
            allowed_tools=state.turn_intent.get("allowed_tools", []),
            db_path=db_path,
        )
        state.used_tools = _ordered_used_tools(state.turn_intent, state.tool_results)
    elif review_mode == "budget_recommendation":
        state.tool_results = _load_budget_recommendation_tool_results(
            user_message,
            allowed_tools=state.turn_intent.get("allowed_tools", []),
            db_path=db_path,
        )
        state.used_tools = _ordered_used_tools(state.turn_intent, state.tool_results)

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
        "turn_intent": state.turn_intent,
        "tool_results": state.tool_results,
        "prompt_context": state.prompt_context,
        "model_response": state.model_response,
        "active_budget_plan": state.active_budget_plan,
        "budget_status": state.budget_status,
    }


def _ordered_used_tools(turn_intent: dict[str, Any], tool_results: dict[str, Any]) -> list[str]:
    """Prefer the interpreted allowed-tool order while keeping any actual tool results."""
    ordered = [tool_name for tool_name in turn_intent.get("allowed_tools", []) if tool_name in tool_results]
    extras = [tool_name for tool_name in tool_results.keys() if tool_name not in ordered]
    return ordered + extras


def _filter_allowed_tool_arguments(
    tool_arguments: dict[str, dict[str, Any]],
    *,
    allowed_tools: list[str],
    workflow_name: str,
) -> dict[str, dict[str, Any]]:
    """Restrict a workflow to the explicit tool subset allowed for this turn."""
    if not allowed_tools:
        raise ValueError(f"No allowed tools were provided for the {workflow_name} workflow.")

    allowed_argument_map = {
        tool_name: arguments
        for tool_name, arguments in tool_arguments.items()
        if tool_name in allowed_tools
    }
    if not allowed_argument_map:
        raise ValueError(
            f"No allowed tools matched the supported tool set for the {workflow_name} workflow."
        )
    return allowed_argument_map


def _load_historical_tool_results(
    user_message: str,
    *,
    allowed_tools: list[str],
    db_path: Optional[str] = None,
) -> dict[str, Any]:
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
    allowed_argument_map = _filter_allowed_tool_arguments(
        tool_arguments,
        allowed_tools=allowed_tools,
        workflow_name="historical_review",
    )
    return {
        tool_name: call_tool_payload(tool_name, arguments=arguments, db_path=db_path)
        for tool_name, arguments in allowed_argument_map.items()
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


def _load_budget_recommendation_tool_results(
    user_message: str,
    *,
    allowed_tools: list[str],
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load the recommendation tool payload for budget-creation style requests."""
    period_start, period_end = _resolve_budget_recommendation_window(user_message)
    savings_target = _extract_savings_target(user_message)
    tool_arguments = {
        "recommend_budget_targets": {
            "period_start": period_start,
            "period_end": period_end,
            "history_periods": 3,
            "savings_target": savings_target,
        }
    }
    allowed_argument_map = _filter_allowed_tool_arguments(
        tool_arguments,
        allowed_tools=allowed_tools,
        workflow_name="budget_recommendation",
    )
    return {
        tool_name: call_tool_payload(tool_name, arguments=arguments, db_path=db_path)
        for tool_name, arguments in allowed_argument_map.items()
    }


def _resolve_budget_recommendation_window(user_message: str) -> tuple[str, str]:
    """Resolve a narrow set of supported budget recommendation windows."""
    normalized = user_message.strip().lower()
    today = date.today()
    if "starting today" in normalized and "month" in normalized:
        end = today + timedelta(days=29)
        return today.isoformat(), end.isoformat()
    if "next month" in normalized:
        next_month_start = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        next_month_end = ((next_month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1))
        return next_month_start.isoformat(), next_month_end.isoformat()
    end = today + timedelta(days=29)
    return today.isoformat(), end.isoformat()


def _extract_savings_target(user_message: str) -> float | None:
    """Extract a simple savings target like 'save $500' from the user message."""
    match = re.search(r"save\s+\$?(\d+(?:\.\d+)?)", user_message.lower())
    if not match:
        return None
    return round(float(match.group(1)), 2)


def _previous_matching_window(period_start: str, period_end: str) -> tuple[str, str]:
    """Return the previous window with the same inclusive duration for drift comparisons."""
    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    duration_days = (end - start).days + 1
    baseline_end = start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=duration_days - 1)
    return baseline_start.isoformat(), baseline_end.isoformat()
