import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app
from backend.services.chat import ChatContext, ChatRequest, ChatResponse, ChatSource, generate_chat_response


def _stub_retrieval_pack(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": ["2026-03-16"],
            "amount": [-42.5],
            "payee": ["Coffee Shop"],
            "category_name": ["Dining"],
            "account_name": ["Visa"],
        }
    )

    monkeypatch.setattr("backend.services.chat.get_transactions_in_date_range", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(
        "backend.services.chat.get_week_rollups",
        lambda *args, **kwargs: {
            "window": {"start": "2026-03-16", "end": "2026-03-22"},
            "summary": {"total_income": 1200.0, "total_expense": 42.5, "net_cashflow": 1157.5},
            "by_category": [{"category": "Dining", "amount": 42.5}],
            "top_payees": [{"payee": "Coffee Shop", "amount": 42.5}],
        },
    )
    monkeypatch.setattr(
        "backend.services.chat.compare_week_over_week",
        lambda *args, **kwargs: {
            "totals": {
                "this_week": {"income": 1200.0, "expense": 42.5, "net": 1157.5},
                "last_week": {"income": 1100.0, "expense": 30.0, "net": 1070.0},
                "delta": {"income": 100.0, "expense": 12.5, "net": 87.5},
                "pct_change": {"income": 9.09, "expense": 41.67, "net": 8.18},
            },
            "category_changes": [],
        },
    )
    monkeypatch.setattr("backend.services.chat.search_past_weeks_by_category", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.services.chat.find_similar_spending_weeks", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.services.chat.get_recent_anomalies", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.services.chat.search_reports", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.services.chat.search_documents", lambda *args, **kwargs: [])


def test_generate_chat_response_returns_structured_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _stub_retrieval_pack(monkeypatch)

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
    assert response.sources
    assert response.actions == ["Compare to last week", "Show similar weeks", "Search historical reports"]
    assert response.retrieval_strategy == ["live_rollup", "week_over_week"]


def test_generate_chat_response_falls_back_when_model_call_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_retrieval_pack(monkeypatch)

    class FailingChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("backend.services.chat.ChatOpenAI", FailingChatOpenAI)

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


def test_chat_endpoint_uses_chat_service(monkeypatch):
    expected = ChatResponse(
        conversation_id="conv-123",
        content="### Visa\n\nHere is a grounded answer.",
        sources=[ChatSource(label="Current window", detail="2026-03-16 to 2026-03-22")],
        actions=["Show similar weeks"],
        facts={"window": {"start": "2026-03-16", "end": "2026-03-22"}},
        retrieval_strategy=["live_rollup"],
    )
    monkeypatch.setattr("backend.app.generate_chat_response", lambda request: expected)

    client = TestClient(app)
    response = client.post(
        "/api/chat",
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
