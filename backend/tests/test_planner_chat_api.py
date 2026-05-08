import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app
from backend.services.conversations import load_conversation
from backend.services.planner_chat import PlannerChatRequest, generate_planner_chat_response


def test_generate_planner_chat_response_persists_pending_recommendation(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "planner-chat.sqlite"))

    monkeypatch.setattr(
        "backend.services.planner_chat.run_planner_agent_turn",
        lambda user_message, planner_state=None, db_path=None: {
            "summary": "Draft ready.",
            "highlights": ["Bills recommended at $1200."],
            "next_action": "Review the draft and approve it when ready.",
            "used_tools": ["recommend_budget_targets"],
            "turn_intent": {"intent": "budget_recommendation"},
            "tool_results": {
                "recommend_budget_targets": {
                    "period_start": "2026-05-06",
                    "period_end": "2026-06-04",
                    "planned_savings": 500.0,
                    "total_budgeted_spend": 2413.0,
                    "buffer_remaining": 87.0,
                    "category_targets": [
                        {"category_name": "Bills", "recommended_target": 1200.0},
                        {"category_name": "Grocery", "recommended_target": 483.0},
                    ],
                }
            },
            "updated_planner_state": {
                "assistant_mode": "planner",
                "awaiting_approval": True,
                "pending_recommendation": {
                    "period_start": "2026-05-06",
                    "period_end": "2026-06-04",
                    "planned_savings": 500.0,
                    "category_targets": [
                        {"category_name": "Bills", "recommended_target": 1200.0},
                        {"category_name": "Grocery", "recommended_target": 483.0},
                    ],
                },
                "last_create_payload": None,
                "latest_saved_plan": None,
            },
        },
    )

    response = generate_planner_chat_response(
        PlannerChatRequest(
            message="Create a budget starting today for a month and save $500",
            conversation_id="planner-conv-1",
            context={
                "account_pid": "acct-123",
                "account_name": "Visa",
                "card_label": "Visa",
                "selected_tab": "budget",
            },
        )
    )

    assert response.conversation_id == "planner-conv-1"
    assert response.planner_state["awaiting_approval"] is True
    assert "Draft ready." in response.content
    assert "**Proposed Budget**" in response.content
    assert "| Bills | $1,200.00 |" in response.content
    assert "- Planned savings: $500.00" in response.content

    thread = load_conversation("planner-conv-1", db_path=str(tmp_path / "planner-chat.sqlite"))
    assert thread["account_pid"] == "acct-123"
    assert thread["context"]["planner_state"]["awaiting_approval"] is True
    assert len(thread["messages"]) == 2
    assert thread["messages"][0]["role"] == "user"
    assert thread["messages"][1]["role"] == "assistant"


def test_generate_planner_chat_response_loads_existing_planner_state_for_follow_up(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "planner-chat.sqlite"))

    seen_states = []

    def fake_turn_runner(user_message, planner_state=None, db_path=None):
        seen_states.append(dict(planner_state or {}))
        if user_message.startswith("Create"):
            return {
                "summary": "Draft ready.",
                "highlights": ["Initial draft prepared."],
                "next_action": "Review or revise the draft.",
                "used_tools": ["recommend_budget_targets"],
                "turn_intent": {"intent": "budget_recommendation"},
                "updated_planner_state": {
                    "assistant_mode": "planner",
                    "awaiting_approval": True,
                    "pending_recommendation": {
                        "period_start": "2026-05-06",
                        "period_end": "2026-06-04",
                        "planned_savings": 500.0,
                    },
                    "last_create_payload": None,
                    "latest_saved_plan": None,
                },
            }
        return {
            "summary": "Budget saved.",
            "highlights": ["Savings saved at $500."],
            "next_action": "Review the new active plan and monitor it.",
            "used_tools": [
                "prepare_budget_plan_from_recommendation",
                "create_budget_plan",
            ],
            "turn_intent": {"intent": "budget_approval"},
            "updated_planner_state": {
                "assistant_mode": "planner",
                "awaiting_approval": False,
                "pending_recommendation": None,
                "last_create_payload": {"status": "active"},
                "latest_saved_plan": {"plan_id": "plan-1"},
            },
        }

    monkeypatch.setattr("backend.services.planner_chat.run_planner_agent_turn", fake_turn_runner)

    first = generate_planner_chat_response(
        PlannerChatRequest(
            message="Create a budget starting today for a month and save $500",
            conversation_id="planner-conv-2",
            context={"account_pid": "acct-123", "selected_tab": "budget"},
        )
    )
    second = generate_planner_chat_response(
        PlannerChatRequest(
            message="Approve this budget",
            conversation_id="planner-conv-2",
            context={"account_pid": "acct-123", "selected_tab": "budget"},
        )
    )

    assert first.planner_state["awaiting_approval"] is True
    assert seen_states[1]["pending_recommendation"]["planned_savings"] == 500.0
    assert second.planner_state["awaiting_approval"] is False
    assert second.planner_state["latest_saved_plan"]["plan_id"] == "plan-1"


def test_generate_planner_chat_response_renders_saved_budget_details(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCE_CHAT_DB_PATH", str(tmp_path / "planner-chat.sqlite"))

    monkeypatch.setattr(
        "backend.services.planner_chat.run_planner_agent_turn",
        lambda user_message, planner_state=None, db_path=None: {
            "summary": "Budget saved.",
            "highlights": ["Savings saved at $500."],
            "next_action": "Review the new active plan and monitor it.",
            "used_tools": [
                "prepare_budget_plan_from_recommendation",
                "create_budget_plan",
            ],
            "turn_intent": {"intent": "budget_approval"},
            "tool_results": {
                "create_budget_plan": {
                    "period_start": "2026-05-06",
                    "period_end": "2026-06-04",
                    "targets": [
                        {"category_name": "Bills", "target_amount": 1200.0},
                        {"category_name": "Savings", "target_amount": 500.0},
                    ],
                }
            },
            "updated_planner_state": {
                "assistant_mode": "planner",
                "awaiting_approval": False,
                "pending_recommendation": None,
                "last_create_payload": None,
                "latest_saved_plan": {
                    "plan_id": "plan-1",
                    "period_start": "2026-05-06",
                    "period_end": "2026-06-04",
                    "targets": [
                        {"category_name": "Bills", "target_amount": 1200.0},
                        {"category_name": "Savings", "target_amount": 500.0},
                    ],
                },
            },
        },
    )

    response = generate_planner_chat_response(
        PlannerChatRequest(
            message="Approve this budget",
            conversation_id="planner-conv-4",
            context={"account_pid": "acct-123", "selected_tab": "budget"},
        )
    )

    assert "**Saved Budget**" in response.content
    assert "| Bills | $1,200.00 |" in response.content
    assert "| Savings | $500.00 |" in response.content


def test_planner_chat_endpoint_uses_planner_service(monkeypatch):
    expected = {
        "conversation_id": "planner-conv-3",
        "content": "Budget saved.",
        "summary": "Budget saved.",
        "highlights": ["Savings saved at $500."],
        "next_action": "Review the active plan.",
        "used_tools": ["create_budget_plan"],
        "turn_intent": {"intent": "budget_approval"},
        "planner_state": {"awaiting_approval": False},
    }
    monkeypatch.setattr("backend.app.generate_planner_chat_response", lambda request: expected)

    client = TestClient(app)
    response = client.post(
        "/api/planner/chat",
        json={
            "message": "Approve this budget",
            "conversation_id": "planner-conv-3",
            "context": {"account_pid": "acct-123", "selected_tab": "budget"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "planner-conv-3"
    assert payload["used_tools"] == ["create_budget_plan"]
    assert payload["planner_state"]["awaiting_approval"] is False
