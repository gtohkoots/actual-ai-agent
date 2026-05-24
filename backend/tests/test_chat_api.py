import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app
from backend.services.chat import ChatContext, ChatRequest, ChatResponse, ChatSource, generate_chat_response
from backend.services.conversations import load_conversation


def _stub_analysis_agent(
    monkeypatch,
    *,
    content="### Visa\n\nGrounded fallback.",
    sources=None,
    actions=None,
    facts=None,
    retrieval_strategy=None,
):
    monkeypatch.setattr(
        "backend.services.chat.run_analysis_agent_turn",
        lambda request: {
            "content": content,
            "sources": sources
            or [{"label": "get_portfolio_summary", "detail": "2026-03-16 to 2026-03-22 · Income $1200.00"}],
            "actions": actions or ["Compare to last week", "Review top categories", "Inspect flagged transactions"],
            "facts": facts or {"analysis_request": {"scope": "account", "intent": "comparison"}},
            "retrieval_strategy": retrieval_strategy or ["get_portfolio_summary"],
        },
    )


def test_generate_chat_response_returns_structured_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.analysis.agent.plan_analysis_tool_calls",
        lambda request, analysis_request: {
            "scope": "account",
            "intent": "comparison",
            "tool_calls": [
                {
                    "tool": "compare_periods",
                    "args": {
                        "current_start": "2026-03-16",
                        "current_end": "2026-03-22",
                        "previous_start": "2026-03-09",
                        "previous_end": "2026-03-15",
                    },
                }
            ],
            "planning_mode": "fallback",
            "reasoning": "test",
        },
    )
    monkeypatch.setattr(
        "backend.agents.analysis.agent.execute_analysis_tool_plan",
        lambda tool_plan: {
            "tool_results": [
                {
                    "tool": "compare_periods",
                    "args": tool_plan["tool_calls"][0]["args"],
                    "step": 0,
                    "result": {"total_deltas": {"total_income": 100.0, "total_expense": 12.5, "net_cashflow": 87.5}},
                }
            ],
            "used_tools": ["compare_periods"],
            "failures": [],
        },
    )
    monkeypatch.setattr(
        "backend.agents.analysis.agent.generate_analysis_response",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    response = generate_chat_response(
        ChatRequest(
            message="What changed on this card?",
            conversation_id="conv-1",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )

    assert response.conversation_id == "conv-1"
    assert "Visa" in response.content
    assert "**Findings**" in response.content
    assert response.sources
    assert response.actions == ["Compare to last week", "Review top categories", "Inspect flagged transactions"]
    assert response.retrieval_strategy == ["compare_periods"]


def test_generate_chat_response_falls_back_when_model_call_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.agents.analysis.agent.plan_analysis_tool_calls",
        lambda request, analysis_request: {
            "scope": "account",
            "intent": "comparison",
            "tool_calls": [{"tool": "get_portfolio_summary", "args": {"period_start": "2026-03-16", "period_end": "2026-03-22"}}],
            "planning_mode": "model",
            "reasoning": "test",
        },
    )
    monkeypatch.setattr(
        "backend.agents.analysis.agent.execute_analysis_tool_plan",
        lambda tool_plan: {
            "tool_results": [
                {
                    "tool": "get_portfolio_summary",
                    "args": tool_plan["tool_calls"][0]["args"],
                    "step": 0,
                    "result": {
                        "period_start": "2026-03-16",
                        "period_end": "2026-03-22",
                        "summary": {"total_income": 1200.0, "total_expense": 42.5, "net_cashflow": 1157.5},
                    },
                }
            ],
            "used_tools": ["get_portfolio_summary"],
            "failures": [],
        },
    )
    monkeypatch.setattr(
        "backend.agents.analysis.agent.generate_analysis_response",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    response = generate_chat_response(
        ChatRequest(
            message="What changed on this card?",
            conversation_id="conv-2",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )

    assert response.conversation_id == "conv-2"
    assert "model request failed" in response.content.lower()
    assert response.sources
    assert response.retrieval_strategy == ["get_portfolio_summary"]


def test_chat_endpoint_uses_chat_service(monkeypatch):
    expected = ChatResponse(
        conversation_id="conv-123",
        content="### Visa\n\nHere is a grounded answer.",
        sources=[ChatSource(label="Current window", detail="2026-03-16 to 2026-03-22")],
        actions=["Show similar weeks"],
        facts={"window": {"start": "2026-03-16", "end": "2026-03-22"}},
        retrieval_strategy=["get_portfolio_summary"],
    )
    monkeypatch.setattr("backend.app.generate_chat_response", lambda request: expected)

    client = TestClient(app)
    response = client.post(
        "/api/analysis/chat",
        json={
            "message": "Summarize this card",
            "conversation_id": "conv-123",
            "context": {"card_label": "Visa", "start_date": "2026-03-16", "end_date": "2026-03-22"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "conv-123"
    assert payload["sources"][0]["label"] == "Current window"
    assert payload["actions"] == ["Show similar weeks"]


def test_generate_chat_response_persists_conversation(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "chat.sqlite"))
    _stub_analysis_agent(monkeypatch)

    response = generate_chat_response(
        ChatRequest(
            message="What changed on this card?",
            conversation_id="conv-persist",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                account_pid="acct-123",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )

    thread = load_conversation(response.conversation_id, db_path=str(tmp_path / "chat.sqlite"))
    assert thread["conversation_id"] == "conv-persist"
    assert thread["account_pid"] == "acct-123"
    assert len(thread["messages"]) == 2
    assert thread["messages"][0]["role"] == "user"
    assert thread["messages"][1]["role"] == "assistant"


def test_chat_conversation_endpoint_returns_thread(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "chat.sqlite"))
    _stub_analysis_agent(monkeypatch)

    generate_chat_response(
        ChatRequest(
            message="What changed on this card?",
            conversation_id="conv-thread",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                account_pid="acct-456",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )

    client = TestClient(app)
    response = client.get("/api/analysis/chat/conversations/conv-thread")

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "conv-thread"
    assert payload["account_pid"] == "acct-456"
    assert len(payload["messages"]) == 2


def test_chat_conversations_endpoint_lists_recent_threads(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "chat.sqlite"))
    _stub_analysis_agent(monkeypatch)

    generate_chat_response(
        ChatRequest(
            message="First question",
            conversation_id="conv-a",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                account_pid="acct-789",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )
    generate_chat_response(
        ChatRequest(
            message="Second question",
            conversation_id="conv-b",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                account_pid="acct-789",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )

    client = TestClient(app)
    response = client.get("/api/analysis/chat/conversations", params={"account_pid": "acct-789", "limit": 5})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["account_pid"] == "acct-789"
    assert payload[0]["message_count"] == 2
    assert payload[0]["preview"]


def test_delete_chat_conversation_removes_thread(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "chat.sqlite"))
    _stub_analysis_agent(monkeypatch)

    generate_chat_response(
        ChatRequest(
            message="Delete me",
            conversation_id="conv-delete",
            context=ChatContext(
                card_label="Visa",
                account_name="Visa",
                account_pid="acct-del",
                start_date="2026-03-16",
                end_date="2026-03-22",
            ),
        )
    )

    client = TestClient(app)
    response = client.delete("/api/analysis/chat/conversations/conv-delete")

    assert response.status_code == 204

    not_found = client.get("/api/analysis/chat/conversations/conv-delete")
    assert not_found.status_code == 404
