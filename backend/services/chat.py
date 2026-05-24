from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.agents.analysis.agent import run_analysis_agent_turn
from backend.services.conversations import append_message


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


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None


class ConversationThread(BaseModel):
    conversation_id: str
    account_pid: Optional[str] = None
    account_name: Optional[str] = None
    card_label: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_message_at: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    request_context = request.context.model_dump()
    append_message(
        conversation_id,
        "user",
        request.message,
        context=request_context,
    )

    turn_request = request.model_copy(update={"conversation_id": conversation_id})
    turn_result = run_analysis_agent_turn(turn_request)
    result = ChatResponse(
        conversation_id=conversation_id,
        content=str(turn_result.get("content", "")),
        sources=[
            ChatSource(label=str(item.get("label", "Source")), detail=str(item.get("detail", "")))
            for item in turn_result.get("sources", [])
            if isinstance(item, dict)
        ],
        actions=[str(item) for item in turn_result.get("actions", []) if str(item).strip()],
        facts=dict(turn_result.get("facts", {})),
        retrieval_strategy=[str(item) for item in turn_result.get("retrieval_strategy", [])],
    )
    append_message(
        conversation_id,
        "assistant",
        result.content,
        context=request_context,
    )
    return result
