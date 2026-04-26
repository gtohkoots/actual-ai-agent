from __future__ import annotations

from typing import Any, Optional

from backend.agents.llm import generate_planner_response
from backend.agents.mcp_client import get_multiple_resource_payloads
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
    state.prompt_context = build_planner_prompt_context(
        user_message=user_message,
        active_budget_plan=active_plan,
        budget_status=current_status,
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
        "prompt_context": state.prompt_context,
        "model_response": state.model_response,
        "active_budget_plan": state.active_budget_plan,
        "budget_status": state.budget_status,
    }
