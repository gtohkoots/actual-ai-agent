from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from backend.agents.llm import (
    generate_planner_response,
    interpret_budget_request_parameters,
    interpret_planner_turn_intent,
)
from backend.agents.mcp_client import call_tool_payload, get_multiple_resource_payloads
from backend.agents.prompts import build_planner_prompt_context
from backend.agents.planner_state import PlannerAgentState


def run_planner_agent(user_message: str, *, db_path: Optional[str] = None) -> dict[str, Any]:
    """Run the planner agent over MCP-backed context and LLM synthesis."""
    return run_planner_agent_turn(user_message, planner_state=None, db_path=db_path)


def run_planner_agent_turn(
    user_message: str,
    *,
    planner_state: dict[str, Any] | None = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run one planner-agent turn, consuming and updating planner workflow state."""
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
        planner_state=_normalize_planner_state(planner_state),
    )
    state.updated_planner_state = dict(state.planner_state)
    state.turn_intent = interpret_planner_turn_intent(
        user_message,
        has_pending_recommendation=bool(state.planner_state.get("pending_recommendation")),
    ) or {"intent": "budget_review", "allowed_tools": [], "confidence": 0.0, "notes": "Fallback intent."}
    review_mode = state.turn_intent.get("intent", "budget_review")
    active_plan_revision_seed = None
    if (
        review_mode in {"budget_review", "budget_revision"}
        and not state.planner_state.get("pending_recommendation")
        and _should_revise_active_budget(user_message, active_plan)
    ):
        active_plan_revision_seed = _recommendation_from_active_plan(active_plan)
        review_mode = "budget_revision"
        state.turn_intent = {
            "intent": "budget_revision",
            "confidence": max(float(state.turn_intent.get("confidence", 0.0)), 0.9),
            "needs_pending_recommendation": True,
            "allowed_tools": ["revise_budget_recommendation"],
            "notes": "The user is revising the current active budget.",
        }
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
        state.updated_planner_state = _planner_state_with_pending_recommendation(
            state.planner_state,
            state.tool_results.get("recommend_budget_targets"),
        )
    elif review_mode == "budget_revision":
        pending_recommendation = state.planner_state.get("pending_recommendation") or active_plan_revision_seed
        if not pending_recommendation:
            state.model_response = _missing_pending_recommendation_response("revise")
            state.summary = state.model_response["summary"]
            state.highlights = state.model_response["highlights"]
            state.next_action = state.model_response["next_action"]
            return _serialize_state(state)
        state.tool_results = _load_budget_revision_tool_results(
            user_message,
            pending_recommendation=pending_recommendation,
            allowed_tools=state.turn_intent.get("allowed_tools", []),
            db_path=db_path,
        )
        state.used_tools = _ordered_used_tools(state.turn_intent, state.tool_results)
        state.updated_planner_state = _planner_state_with_pending_recommendation(
            state.planner_state,
            state.tool_results.get("revise_budget_recommendation"),
        )
    elif review_mode == "budget_approval":
        if not state.planner_state.get("pending_recommendation"):
            state.model_response = _missing_pending_recommendation_response("approve")
            state.summary = state.model_response["summary"]
            state.highlights = state.model_response["highlights"]
            state.next_action = state.model_response["next_action"]
            return _serialize_state(state)
        state.tool_results = _load_budget_approval_tool_results(
            pending_recommendation=state.planner_state["pending_recommendation"],
            allowed_tools=state.turn_intent.get("allowed_tools", []),
            db_path=db_path,
        )
        state.used_tools = _ordered_used_tools(state.turn_intent, state.tool_results)
        state.updated_planner_state = _planner_state_after_save(
            state.planner_state,
            create_payload=state.tool_results.get("prepare_budget_plan_from_recommendation"),
            saved_plan=state.tool_results.get("create_budget_plan"),
        )

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
        "planner_state": state.planner_state,
        "updated_planner_state": state.updated_planner_state,
        "tool_results": state.tool_results,
        "prompt_context": state.prompt_context,
        "model_response": state.model_response,
        "active_budget_plan": state.active_budget_plan,
        "budget_status": state.budget_status,
    }


def _normalize_planner_state(planner_state: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize planner workflow state into a stable shape for turn handling."""
    state = dict(planner_state or {})
    return {
        "assistant_mode": str(state.get("assistant_mode") or "planner"),
        "awaiting_approval": bool(state.get("awaiting_approval", False)),
        "pending_recommendation": state.get("pending_recommendation"),
        "last_create_payload": state.get("last_create_payload"),
        "latest_saved_plan": state.get("latest_saved_plan"),
    }


def _planner_state_with_pending_recommendation(
    planner_state: dict[str, Any],
    recommendation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return planner state updated with a recommendation that now awaits approval."""
    updated = dict(_normalize_planner_state(planner_state))
    updated["assistant_mode"] = "planner"
    updated["awaiting_approval"] = recommendation is not None
    updated["pending_recommendation"] = recommendation
    updated["last_create_payload"] = None
    return updated


def _planner_state_after_save(
    planner_state: dict[str, Any],
    *,
    create_payload: dict[str, Any] | None,
    saved_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return planner state updated after an approved draft has been saved."""
    updated = dict(_normalize_planner_state(planner_state))
    updated["assistant_mode"] = "planner"
    updated["awaiting_approval"] = False
    updated["pending_recommendation"] = None
    updated["last_create_payload"] = create_payload
    updated["latest_saved_plan"] = saved_plan
    return updated


def _missing_pending_recommendation_response(action: str) -> dict[str, Any]:
    """Return a deterministic response when revise/approve is requested without a draft."""
    verb = "revise" if action == "revise" else "approve"
    return {
        "summary": f"There is no pending budget draft to {verb} yet.",
        "highlights": ["Start by asking for a budget recommendation before continuing this workflow."],
        "next_action": "Ask for a new budget draft first, then revise or approve it once a recommendation exists.",
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
    request_parameters = _resolve_budget_recommendation_parameters(user_message)
    tool_arguments = {
        "recommend_budget_targets": {
            "period_start": request_parameters["period_start"],
            "period_end": request_parameters["period_end"],
            "history_periods": 3,
            "savings_target": request_parameters["savings_target"],
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


def _load_budget_revision_tool_results(
    user_message: str,
    *,
    pending_recommendation: dict[str, Any],
    allowed_tools: list[str],
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load the revision tool payload for follow-up changes to a pending draft."""
    tool_arguments = {
        "revise_budget_recommendation": {
            "current_recommendation": pending_recommendation,
            "user_comment": user_message,
        }
    }
    allowed_argument_map = _filter_allowed_tool_arguments(
        tool_arguments,
        allowed_tools=allowed_tools,
        workflow_name="budget_revision",
    )
    return {
        tool_name: call_tool_payload(tool_name, arguments=arguments, db_path=db_path)
        for tool_name, arguments in allowed_argument_map.items()
    }


def _load_budget_approval_tool_results(
    *,
    pending_recommendation: dict[str, Any],
    allowed_tools: list[str],
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load the mapper and create-tool payloads for saving an approved budget draft."""
    required_tools = [
        "prepare_budget_plan_from_recommendation",
        "create_budget_plan",
    ]
    if any(tool_name not in allowed_tools for tool_name in required_tools):
        raise ValueError("The budget_approval workflow requires mapper and create tools in the allowed tool set.")

    create_payload = call_tool_payload(
        "prepare_budget_plan_from_recommendation",
        arguments={"recommendation": pending_recommendation},
        db_path=db_path,
    )
    saved_plan = call_tool_payload(
        "create_budget_plan",
        arguments=create_payload,
        db_path=db_path,
    )
    return {
        "prepare_budget_plan_from_recommendation": create_payload,
        "create_budget_plan": saved_plan,
    }


def _resolve_budget_recommendation_parameters(user_message: str) -> dict[str, Any]:
    """Interpret and validate budget recommendation parameters from a natural-language request."""
    request_parameters = interpret_budget_request_parameters(user_message)
    period_start = str(request_parameters.get("period_start", "")).strip()
    period_end = str(request_parameters.get("period_end", "")).strip()
    if not period_start or not period_end:
        raise ValueError("Budget recommendation requires non-empty period_start and period_end.")

    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    if start > end:
        raise ValueError("Budget recommendation period_start cannot be after period_end.")

    savings_target = request_parameters.get("savings_target")
    if savings_target is not None:
        savings_target = round(float(savings_target), 2)
        if savings_target < 0:
            raise ValueError("Budget recommendation savings_target cannot be negative.")

    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "savings_target": savings_target,
    }


def _should_revise_active_budget(user_message: str, active_plan: dict[str, Any]) -> bool:
    """Return whether the user is asking to revise the current active budget in place."""
    if not active_plan or active_plan.get("status") == "missing":
        return False

    normalized = user_message.strip().lower()
    budget_reference_markers = [
        "current budget",
        "active budget",
        "my budget",
    ]
    revision_markers = [
        "update",
        "revise",
        "adjust",
        "change",
        "increase ",
        "decrease ",
        "reduce ",
        "raise ",
        "lower ",
        "don't touch",
        "do not touch",
        "keep ",
        "instead",
    ]
    return any(marker in normalized for marker in budget_reference_markers) and any(
        marker in normalized for marker in revision_markers
    )


def _recommendation_from_active_plan(active_plan: dict[str, Any]) -> dict[str, Any]:
    """Convert the current active saved budget into a revise-compatible recommendation payload."""
    targets = list(active_plan.get("targets", []) or [])
    savings_target = 0.0
    category_targets: list[dict[str, Any]] = []
    for item in targets:
        category_name = str(item.get("category_name", "")).strip()
        if not category_name:
            continue
        target_amount = round(float(item.get("target_amount", 0.0)), 2)
        if category_name == "Savings":
            savings_target = target_amount
            continue
        category_targets.append(
            {
                "category_name": category_name,
                "recommended_target": target_amount,
                "baseline_amount": target_amount,
            }
        )

    total_budgeted_spend = round(sum(item["recommended_target"] for item in category_targets), 2)
    return {
        "period_start": str(active_plan.get("period_start", "")).strip(),
        "period_end": str(active_plan.get("period_end", "")).strip(),
        "planned_savings": savings_target,
        "total_budgeted_spend": total_budgeted_spend,
        "category_targets": category_targets,
        "assumptions": {
            "history_periods": 3,
            "source": "active_budget_plan",
        },
    }


def _previous_matching_window(period_start: str, period_end: str) -> tuple[str, str]:
    """Return the previous window with the same inclusive duration for drift comparisons."""
    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    duration_days = (end - start).days + 1
    baseline_end = start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=duration_days - 1)
    return baseline_start.isoformat(), baseline_end.isoformat()
