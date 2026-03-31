from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.services.dashboard import DashboardOverview, list_accounts, build_dashboard_overview
from backend.services.chat import ChatRequest, ChatResponse, ConversationThread, generate_chat_response
from backend.services.conversations import delete_conversation, list_conversations, load_conversation
from backend.services.documents import rebuild_document_store, search_documents


def _cors_origins() -> List[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173")
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(title="Finance Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/accounts")
def accounts() -> list[dict]:
    return list_accounts()


@app.get("/api/dashboard", response_model=DashboardOverview)
def dashboard(start_date: str | None = None, end_date: str | None = None) -> DashboardOverview:
    return build_dashboard_overview(start_date=start_date, end_date=end_date)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return generate_chat_response(request)


@app.get("/api/chat/conversations/{conversation_id}", response_model=ConversationThread)
def chat_conversation(conversation_id: str) -> ConversationThread:
    try:
        return ConversationThread.model_validate(load_conversation(conversation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@app.get("/api/chat/conversations")
def chat_conversations(account_pid: str | None = None, limit: int = 8) -> list[dict]:
    return list_conversations(account_pid=account_pid, limit=limit)


@app.delete("/api/chat/conversations/{conversation_id}", status_code=204)
def delete_chat_conversation(conversation_id: str) -> None:
    delete_conversation(conversation_id)


@app.post("/api/documents/rebuild")
def rebuild_documents() -> dict:
    return rebuild_document_store(".")


@app.get("/api/documents/search")
def documents_search(query: str, limit: int = 5) -> list[dict]:
    return search_documents(query=query, limit=limit)
