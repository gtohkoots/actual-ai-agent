from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.agents.planner_agent import run_planner_agent_turn
from backend.services.conversations import append_message, load_planner_state, save_planner_state


class PlannerChatMessageInput(BaseModel):
    role: str = Field(..., description="system, user, or assistant")
    content: str = Field(..., description="Message content")


class PlannerChatContext(BaseModel):
    selected_tab: Optional[str] = Field(None, description="Selected UI tab")
    account_pid: Optional[str] = Field(None, description="Actual account pid")
    account_name: Optional[str] = Field(None, description="Actual account name")
    card_label: Optional[str] = Field(None, description="Displayed card label")
    start_date: Optional[str] = Field(None, description="Window start YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="Window end YYYY-MM-DD")


class PlannerChatRequest(BaseModel):
    message: str = Field(..., description="Latest user message")
    conversation_id: Optional[str] = Field(None, description="Client conversation id")
    history: List[PlannerChatMessageInput] = Field(default_factory=list, description="Recent conversation turns")
    context: PlannerChatContext = Field(default_factory=PlannerChatContext)


class PlannerChatResponse(BaseModel):
    conversation_id: str
    content: str
    summary: str
    highlights: List[str] = Field(default_factory=list)
    next_action: str
    used_tools: List[str] = Field(default_factory=list)
    turn_intent: Dict[str, Any] = Field(default_factory=dict)
    planner_state: Dict[str, Any] = Field(default_factory=dict)


def generate_planner_chat_response(request: PlannerChatRequest) -> PlannerChatResponse:
    """Run one planner chat turn while persisting messages and planner workflow state."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    request_context = request.context.model_dump(exclude_none=True)

    append_message(
        conversation_id,
        "user",
        request.message,
        context=request_context,
    )

    planner_state = load_planner_state(conversation_id)
    turn_result = run_planner_agent_turn(
        request.message,
        planner_state=planner_state,
    )
    updated_planner_state = turn_result.get("updated_planner_state", planner_state)
    save_planner_state(conversation_id, updated_planner_state)

    content = _render_planner_chat_content(turn_result)
    append_message(
        conversation_id,
        "assistant",
        content,
        context=request_context,
    )

    return PlannerChatResponse(
        conversation_id=conversation_id,
        content=content,
        summary=str(turn_result.get("summary", "")),
        highlights=[str(item) for item in turn_result.get("highlights", [])],
        next_action=str(turn_result.get("next_action", "")),
        used_tools=[str(item) for item in turn_result.get("used_tools", [])],
        turn_intent=dict(turn_result.get("turn_intent", {})),
        planner_state=dict(updated_planner_state or {}),
    )


def _render_planner_chat_content(turn_result: dict[str, Any]) -> str:
    """Render a frontend-friendly markdown reply from a planner turn result."""
    summary = str(turn_result.get("summary", "")).strip()
    highlights = [str(item).strip() for item in turn_result.get("highlights", []) if str(item).strip()]
    next_action = str(turn_result.get("next_action", "")).strip()
    plan_details = _render_budget_plan_details(turn_result)

    sections: list[str] = []
    if summary:
        sections.append(summary)
    if plan_details:
        sections.append(plan_details)
    if highlights:
        sections.append("**Highlights**\n" + "\n".join(f"- {item}" for item in highlights))
    if next_action:
        sections.append(f"**Next**\n{next_action}")
    return "\n\n".join(sections) if sections else "No planner response was generated."


def _render_budget_plan_details(turn_result: dict[str, Any]) -> str:
    """Render detailed budget targets when the turn produced or saved a budget plan."""
    turn_intent = dict(turn_result.get("turn_intent", {}))
    intent = str(turn_intent.get("intent", "")).strip()
    tool_results = dict(turn_result.get("tool_results", {}))
    planner_state = dict(turn_result.get("updated_planner_state", turn_result.get("planner_state", {})) or {})

    if intent == "budget_recommendation":
        plan = dict(
            tool_results.get("recommend_budget_targets")
            or planner_state.get("pending_recommendation")
            or {}
        )
        return _render_recommendation_details(plan, heading="Proposed Budget")

    if intent == "budget_revision":
        plan = dict(
            tool_results.get("revise_budget_recommendation")
            or planner_state.get("pending_recommendation")
            or {}
        )
        return _render_recommendation_details(plan, heading="Updated Budget")

    if intent == "budget_approval":
        plan = dict(
            tool_results.get("create_budget_plan")
            or tool_results.get("prepare_budget_plan_from_recommendation")
            or planner_state.get("latest_saved_plan")
            or planner_state.get("last_create_payload")
            or {}
        )
        return _render_saved_plan_details(plan)

    return ""


def _render_recommendation_details(plan: dict[str, Any], *, heading: str) -> str:
    """Render a recommendation payload into a readable markdown plan section."""
    if not plan:
        return ""

    lines: list[str] = [f"**{heading}**"]
    period_start = str(plan.get("period_start", "")).strip()
    period_end = str(plan.get("period_end", "")).strip()
    if period_start and period_end:
        lines.append(f"- Period: {period_start} to {period_end}")

    planned_savings = plan.get("planned_savings")
    if isinstance(planned_savings, (int, float)):
        lines.append(f"- Planned savings: {_format_currency(planned_savings)}")

    total_budgeted_spend = plan.get("total_budgeted_spend")
    if isinstance(total_budgeted_spend, (int, float)):
        lines.append(f"- Total budgeted spend: {_format_currency(total_budgeted_spend)}")

    buffer_remaining = plan.get("buffer_remaining")
    if isinstance(buffer_remaining, (int, float)):
        lines.append(f"- Buffer remaining: {_format_currency(buffer_remaining)}")

    category_targets = [
        item for item in plan.get("category_targets", [])
        if isinstance(item, dict) and isinstance(item.get("recommended_target"), (int, float))
    ]
    if category_targets:
        lines.append("")
        lines.append("| Category | Target |")
        lines.append("| --- | ---: |")
        for item in category_targets:
            lines.append(
                f'| {item.get("category_name", "Uncategorized")} | '
                f'{_format_currency(item["recommended_target"])} |'
            )

    return "\n".join(lines)


def _render_saved_plan_details(plan: dict[str, Any]) -> str:
    """Render the saved plan payload into a readable markdown plan section."""
    if not plan:
        return ""

    lines: list[str] = ["**Saved Budget**"]
    period_start = str(plan.get("period_start", "")).strip()
    period_end = str(plan.get("period_end", "")).strip()
    if period_start and period_end:
        lines.append(f"- Period: {period_start} to {period_end}")

    targets = [
        item for item in plan.get("targets", [])
        if isinstance(item, dict) and isinstance(item.get("target_amount"), (int, float))
    ]
    if targets:
        lines.append("")
        lines.append("| Category | Target |")
        lines.append("| --- | ---: |")
        for item in targets:
            lines.append(
                f'| {item.get("category_name", "Uncategorized")} | '
                f'{_format_currency(item["target_amount"])} |'
            )

    return "\n".join(lines)


def _format_currency(value: float) -> str:
    """Format a numeric amount as USD currency."""
    return f"${value:,.2f}"
