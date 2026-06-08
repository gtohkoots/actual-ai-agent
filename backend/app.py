from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.services.dashboard import DashboardOverview, list_accounts, build_dashboard_overview
from backend.services.chat import ChatRequest, ChatResponse, ConversationThread, generate_chat_response
from backend.services.analysis_options import list_analysis_categories, list_analysis_payees
from backend.services.conversations import delete_conversation, list_conversations, load_conversation
from backend.services.documents import rebuild_document_store, search_documents
from backend.services.investments import (
    get_investment_industry_exposure,
    get_investments_overview,
    get_single_name_exposure,
    import_fidelity_positions_csv,
    refresh_investment_industry_exposure,
    refresh_investment_fund_holdings,
)
from backend.services.planner_chat import PlannerChatRequest, PlannerChatResponse, generate_planner_chat_response
from backend.services.planner_overview import PlannerOverviewResponse, generate_planner_overview

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


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


@app.post("/api/analysis/chat", response_model=ChatResponse)
def analysis_chat(request: ChatRequest) -> ChatResponse:
    return generate_chat_response(request)


@app.post("/api/planner/chat", response_model=PlannerChatResponse)
def planner_chat(request: PlannerChatRequest) -> PlannerChatResponse:
    return generate_planner_chat_response(request)


@app.get("/api/planner/overview", response_model=PlannerOverviewResponse)
def planner_overview() -> PlannerOverviewResponse:
    return generate_planner_overview()


@app.get("/api/investments/overview")
def investments_overview() -> dict:
    return get_investments_overview()


@app.post("/api/investments/import/fidelity-positions-csv")
def import_investment_positions_csv(
    payload: bytes = Body(..., media_type="text/csv"),
    filename: str | None = None,
    as_of_date: str | None = None,
) -> dict:
    try:
        return import_fidelity_positions_csv(payload, filename=filename, as_of_date=as_of_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/investments/exposure/single-name")
def investment_single_name_exposure(min_percent: float = 1.0, limit: int = 15) -> dict:
    result = get_single_name_exposure(min_percent=min_percent, limit=limit)
    logger.info(
        "Investment exposure response status=%s items=%s unresolved=%s summary=%s",
        result.get("status"),
        len(result.get("items") or []),
        len(result.get("unresolved") or []),
        result.get("summary") or {},
    )
    return result


@app.post("/api/investments/exposure/refresh")
def refresh_investment_exposure(force: bool = False, request_delay_seconds: float = 1.2) -> dict:
    result = refresh_investment_fund_holdings(force=force, request_delay_seconds=request_delay_seconds)
    logger.info(
        "Investment exposure refresh response status=%s candidates=%s refreshed=%s skipped=%s failed=%s",
        result.get("status"),
        result.get("fund_candidates") or [],
        result.get("refreshed") or [],
        result.get("skipped") or [],
        result.get("failed") or [],
    )
    return result


@app.get("/api/investments/industry/exposure")
def investment_industry_exposure() -> dict:
    result = get_investment_industry_exposure()
    logger.info(
        "Investment industry exposure response status=%s items=%s summary=%s",
        result.get("status"),
        len(result.get("items") or []),
        result.get("summary") or {},
    )
    return result


@app.post("/api/investments/industry/refresh")
def refresh_investment_industry(force: bool = False) -> dict:
    result = refresh_investment_industry_exposure(force=force)
    logger.info(
        "Investment industry refresh response status=%s items=%s requests=%s cache_hits=%s",
        result.get("status"),
        len(result.get("items") or []),
        result.get("classification_requests"),
        result.get("classification_cache_hits"),
    )
    return result


@app.get("/api/analysis/options/categories")
def analysis_categories(
    account_pid: str | None = None,
    account_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    return {
        "items": list_analysis_categories(
            account_pid=account_pid,
            account_name=account_name,
            start_date=start_date,
            end_date=end_date,
        )
    }


@app.get("/api/analysis/options/payees")
def analysis_payees(
    account_pid: str | None = None,
    account_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
) -> dict:
    return {
        "items": list_analysis_payees(
            account_pid=account_pid,
            account_name=account_name,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    }


@app.get("/api/analysis/chat/conversations/{conversation_id}", response_model=ConversationThread)
def analysis_chat_conversation(conversation_id: str) -> ConversationThread:
    try:
        return ConversationThread.model_validate(load_conversation(conversation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


@app.get("/api/analysis/chat/conversations")
def analysis_chat_conversations(account_pid: str | None = None, limit: int = 8) -> list[dict]:
    return list_conversations(account_pid=account_pid, limit=limit)


@app.delete("/api/analysis/chat/conversations/{conversation_id}", status_code=204)
def delete_analysis_chat_conversation(conversation_id: str) -> None:
    delete_conversation(conversation_id)


@app.post("/api/documents/rebuild")
def rebuild_documents() -> dict:
    return rebuild_document_store(".")


@app.get("/api/documents/search")
def documents_search(query: str, limit: int = 5) -> list[dict]:
    return search_documents(query=query, limit=limit)
