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

    sections: list[str] = []
    if summary:
        sections.append(summary)
    if highlights:
        sections.append("**Highlights**\n" + "\n".join(f"- {item}" for item in highlights))
    if next_action:
        sections.append(f"**Next**\n{next_action}")
    return "\n\n".join(sections) if sections else "No planner response was generated."
