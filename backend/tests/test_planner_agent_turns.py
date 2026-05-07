import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.agents.planner_agent as planner_agent


def _missing_payloads():
    return {
        "planner://budget/active-plan": {"status": "missing"},
        "planner://budget/current-status": {"status": "missing"},
    }


def test_run_planner_agent_turn_stores_pending_recommendation(monkeypatch):
    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: _missing_payloads()[uri] for uri in uris},
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_recommendation",
            "confidence": 0.95,
            "needs_pending_recommendation": False,
            "allowed_tools": ["recommend_budget_targets"],
            "notes": "Budget recommendation request.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "call_tool_payload",
        lambda tool_name, arguments=None, db_path=None: {
            "period_start": "2026-05-06",
            "period_end": "2026-06-04",
            "planned_savings": 500.0,
            "category_targets": [{"category_name": "Bills", "recommended_target": 1200.0}],
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "Draft ready.",
            "highlights": ["Bills was recommended at $1200."],
            "next_action": "Review the draft and confirm whether you want to save it as a new budget plan.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "date",
        type(
            "FakeDate",
            (),
            {
                "today": staticmethod(lambda: __import__("datetime").date(2026, 5, 6)),
                "fromisoformat": staticmethod(__import__("datetime").date.fromisoformat),
            },
        ),
    )

    result = planner_agent.run_planner_agent_turn(
        "Create a budget starting today for a month and save $500",
        planner_state={"assistant_mode": "planner"},
    )

    assert result["used_tools"] == ["recommend_budget_targets"]
    assert result["updated_planner_state"]["awaiting_approval"] is True
    assert result["updated_planner_state"]["pending_recommendation"]["planned_savings"] == 500.0


def test_run_planner_agent_turn_revises_pending_recommendation(monkeypatch):
    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: _missing_payloads()[uri] for uri in uris},
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_revision",
            "confidence": 0.9,
            "needs_pending_recommendation": True,
            "allowed_tools": ["revise_budget_recommendation"],
            "notes": "Revision request.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "call_tool_payload",
        lambda tool_name, arguments=None, db_path=None: {
            "period_start": "2026-05-06",
            "period_end": "2026-06-04",
            "planned_savings": 500.0,
            "category_targets": [{"category_name": "Dine", "recommended_target": 275.0}],
            "revision_context": {"protected_categories": ["Bills"]},
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "Revised draft ready.",
            "highlights": ["Dine was raised.", "Bills stayed protected."],
            "next_action": "Review the revised draft and confirm whether you want to save it.",
        },
    )

    result = planner_agent.run_planner_agent_turn(
        "Increase Dine a bit and keep Bills fixed.",
        planner_state={
            "assistant_mode": "planner",
            "awaiting_approval": True,
            "pending_recommendation": {
                "period_start": "2026-05-06",
                "period_end": "2026-06-04",
                "planned_savings": 500.0,
                "category_targets": [{"category_name": "Dine", "recommended_target": 247.0}],
            },
        },
    )

    assert result["used_tools"] == ["revise_budget_recommendation"]
    assert result["updated_planner_state"]["awaiting_approval"] is True
    assert result["updated_planner_state"]["pending_recommendation"]["category_targets"][0]["recommended_target"] == 275.0


def test_run_planner_agent_turn_approves_and_saves_pending_recommendation(monkeypatch):
    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: _missing_payloads()[uri] for uri in uris},
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_approval",
            "confidence": 0.96,
            "needs_pending_recommendation": True,
            "allowed_tools": [
                "prepare_budget_plan_from_recommendation",
                "create_budget_plan",
            ],
            "notes": "Approval request.",
        },
    )

    def fake_call_tool_payload(tool_name, arguments=None, db_path=None):
        if tool_name == "prepare_budget_plan_from_recommendation":
            return {
                "period_start": "2026-05-06",
                "period_end": "2026-06-04",
                "targets": [
                    {"category_name": "Bills", "target_amount": 1200.0},
                    {"category_name": "Savings", "target_amount": 500.0},
                ],
                "status": "active",
            }
        return {
            "plan_id": "plan-1",
            "period_start": "2026-05-06",
            "period_end": "2026-06-04",
            "status": "active",
            "targets": [
                {"category_name": "Bills", "target_amount": 1200.0},
                {"category_name": "Savings", "target_amount": 500.0},
            ],
        }

    monkeypatch.setattr(planner_agent, "call_tool_payload", fake_call_tool_payload)
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "Budget saved.",
            "highlights": ["Bills saved at $1200.", "Savings saved at $500."],
            "next_action": "Review the new active plan and monitor it against live spending.",
        },
    )

    result = planner_agent.run_planner_agent_turn(
        "Approve this budget.",
        planner_state={
            "assistant_mode": "planner",
            "awaiting_approval": True,
            "pending_recommendation": {
                "period_start": "2026-05-06",
                "period_end": "2026-06-04",
                "planned_savings": 500.0,
                "category_targets": [{"category_name": "Bills", "recommended_target": 1200.0}],
            },
        },
    )

    assert result["used_tools"] == [
        "prepare_budget_plan_from_recommendation",
        "create_budget_plan",
    ]
    assert result["updated_planner_state"]["awaiting_approval"] is False
    assert result["updated_planner_state"]["pending_recommendation"] is None
    assert result["updated_planner_state"]["latest_saved_plan"]["plan_id"] == "plan-1"


def test_run_planner_agent_turn_handles_revision_without_pending_recommendation(monkeypatch):
    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: _missing_payloads()[uri] for uri in uris},
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_revision",
            "confidence": 0.86,
            "needs_pending_recommendation": True,
            "allowed_tools": ["revise_budget_recommendation"],
            "notes": "Revision request.",
        },
    )

    result = planner_agent.run_planner_agent_turn(
        "Raise Dine a bit.",
        planner_state={"assistant_mode": "planner"},
    )

    assert result["used_tools"] == []
    assert result["summary"] == "There is no pending budget draft to revise yet."
    assert result["updated_planner_state"]["pending_recommendation"] is None
