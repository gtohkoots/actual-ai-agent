import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.agents.planner_agent as planner_agent


def test_run_planner_agent_handles_missing_active_budget(monkeypatch):
    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {
            uri: {"status": "missing", "message": "No active budget plan found."} for uri in uris
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "No active budget plan is set up yet.",
            "highlights": ["The planner could not find an active budget plan to review."],
            "next_action": "Create a budget plan before requesting a budget review.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_review",
            "confidence": 0.9,
            "needs_pending_recommendation": False,
            "allowed_tools": [],
            "notes": "Current-budget review request.",
        },
    )

    result = planner_agent.run_planner_agent("Review my budget")

    assert result["summary"] == "No active budget plan is set up yet."
    assert result["next_action"] == "Create a budget plan before requesting a budget review."
    assert result["used_resources"] == [
        "planner://budget/active-plan",
        "planner://budget/current-status",
    ]
    assert result["used_tools"] == []
    assert result["turn_intent"]["intent"] == "budget_review"


def test_run_planner_agent_summarizes_budget_status(monkeypatch):
    payloads = {
        "planner://budget/active-plan": {
            "plan_id": "plan-1",
            "period_start": "2026-04-01",
            "period_end": "2026-04-30",
            "status": "active",
            "targets": [],
        },
        "planner://budget/current-status": {
            "plan_id": "plan-1",
            "period_start": "2026-04-01",
            "period_end": "2026-04-30",
            "status": "active",
            "summary": {
                "total_target": 1000.0,
                "total_actual": 620.0,
                "total_remaining": 380.0,
                "utilization_pct": 62.0,
            },
            "categories": [
                {
                    "category_name": "Dining",
                    "target_amount": 200.0,
                    "actual_amount": 240.0,
                    "remaining_amount": -40.0,
                    "utilization_pct": 120.0,
                    "elapsed_period_pct": 80.0,
                    "status": "overspent",
                },
                {
                    "category_name": "Grocery",
                    "target_amount": 400.0,
                    "actual_amount": 360.0,
                    "remaining_amount": 40.0,
                    "utilization_pct": 90.0,
                    "elapsed_period_pct": 80.0,
                    "status": "at_risk",
                },
            ],
        },
    }
    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: payloads[uri] for uri in uris},
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "April budget is mostly healthy, but Dining is already over target.",
            "highlights": [
                "Dining is overspent by $40.00.",
                "Grocery is at risk with 90.0% of its target already used.",
            ],
            "next_action": "Review the overspent categories first and decide whether to reduce spending or raise their targets.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_review",
            "confidence": 0.95,
            "needs_pending_recommendation": False,
            "allowed_tools": [],
            "notes": "Current-budget review request.",
        },
    )

    result = planner_agent.run_planner_agent("Review my budget")

    assert "Dining is already over target" in result["summary"]
    assert "Dining is overspent by $40.00." in result["highlights"]
    assert "Grocery is at risk with 90.0% of its target already used." in result["highlights"]
    assert result["next_action"] == (
        "Review the overspent categories first and decide whether to reduce spending or raise their targets."
    )
    assert result["prompt_context"]["budget_status"]["summary"]["total_actual"] == 620.0


def test_run_planner_agent_calls_historical_tools_for_last_month(monkeypatch):
    payloads = {
        "planner://budget/active-plan": {"status": "missing"},
        "planner://budget/current-status": {"status": "missing"},
    }
    tool_calls = []

    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: payloads[uri] for uri in uris},
    )

    def fake_call_tool_payload(tool_name, arguments=None, db_path=None):
        tool_calls.append((tool_name, arguments))
        if tool_name == "get_portfolio_summary":
            return {
                "period_start": "2026-03-01",
                "period_end": "2026-03-31",
                "summary": {"total_income": 3000.0, "total_expense": 1200.0, "net_cashflow": 1800.0},
            }
        if tool_name == "get_category_spend":
            return {"categories": [{"category_name": "Grocery", "amount": 400.0, "share_pct": 33.33}]}
        if tool_name == "get_account_breakdown":
            return {"accounts": [{"account_name": "Checking", "expense_amount": 1200.0}]}
        return {"top_category_changes": [{"category_name": "Dining", "delta": 85.0}]}

    monkeypatch.setattr(planner_agent, "call_tool_payload", fake_call_tool_payload)
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "historical_review",
            "confidence": 0.93,
            "needs_pending_recommendation": False,
            "allowed_tools": [
                "get_portfolio_summary",
                "get_category_spend",
                "get_account_breakdown",
                "get_spending_drift",
            ],
            "notes": "Historical review request.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "March spending was concentrated in Grocery and rose in Dining.",
            "highlights": ["Grocery was the top category.", "Dining increased versus the prior month."],
            "next_action": "Review whether Dining should be tightened in your next budget.",
        },
    )
    monkeypatch.setattr(planner_agent, "date", type("FakeDate", (), {"today": staticmethod(lambda: __import__("datetime").date(2026, 4, 26)), "fromisoformat": staticmethod(__import__("datetime").date.fromisoformat)}))

    result = planner_agent.run_planner_agent("Review spending for last month")

    assert result["used_tools"] == [
        "get_portfolio_summary",
        "get_category_spend",
        "get_account_breakdown",
        "get_spending_drift",
    ]
    assert result["prompt_context"]["review_mode"] == "historical_review"
    assert tool_calls[0][0] == "get_portfolio_summary"
    assert tool_calls[0][1]["period_start"] == "2026-03-01"
    assert "March spending" in result["summary"]


def test_run_planner_agent_calls_budget_recommendation_tool(monkeypatch):
    payloads = {
        "planner://budget/active-plan": {"status": "missing"},
        "planner://budget/current-status": {"status": "missing"},
    }
    tool_calls = []

    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: payloads[uri] for uri in uris},
    )

    def fake_call_tool_payload(tool_name, arguments=None, db_path=None):
        tool_calls.append((tool_name, arguments))
        return {
            "period_start": "2026-04-28",
            "period_end": "2026-05-27",
            "planned_savings": 500.0,
            "total_budgeted_spend": 2000.0,
            "category_targets": [
                {
                    "category_name": "Grocery",
                    "baseline_amount": 450.0,
                    "recommended_target": 472.5,
                }
            ],
        }

    monkeypatch.setattr(planner_agent, "call_tool_payload", fake_call_tool_payload)
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "budget_recommendation",
            "confidence": 0.94,
            "needs_pending_recommendation": False,
            "allowed_tools": ["recommend_budget_targets"],
            "notes": "Budget recommendation request.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_budget_request_parameters",
        lambda user_message: {
            "period_start": "2026-04-28",
            "period_end": "2026-05-27",
            "savings_target": 500.0,
            "notes": "Structured recommendation request.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "Here is a one-month budget recommendation starting today with savings reserved first.",
            "highlights": ["Grocery is set slightly above baseline to protect essentials."],
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
                "today": staticmethod(lambda: __import__("datetime").date(2026, 4, 28)),
                "fromisoformat": staticmethod(__import__("datetime").date.fromisoformat),
            },
        ),
    )

    result = planner_agent.run_planner_agent("Create a budget starting today for a month and save $500")

    assert result["used_tools"] == ["recommend_budget_targets"]
    assert result["prompt_context"]["review_mode"] == "budget_recommendation"
    assert tool_calls[0][0] == "recommend_budget_targets"
    assert tool_calls[0][1]["period_start"] == "2026-04-28"
    assert tool_calls[0][1]["period_end"] == "2026-05-27"
    assert tool_calls[0][1]["savings_target"] == 500.0
    assert "one-month budget recommendation" in result["summary"]


def test_run_planner_agent_limits_historical_execution_to_allowed_tools(monkeypatch):
    payloads = {
        "planner://budget/active-plan": {"status": "missing"},
        "planner://budget/current-status": {"status": "missing"},
    }
    tool_calls = []

    monkeypatch.setattr(
        planner_agent,
        "get_multiple_resource_payloads",
        lambda uris, db_path=None: {uri: payloads[uri] for uri in uris},
    )
    monkeypatch.setattr(
        planner_agent,
        "interpret_planner_turn_intent",
        lambda user_message, has_pending_recommendation=False: {
            "intent": "historical_review",
            "confidence": 0.9,
            "needs_pending_recommendation": False,
            "allowed_tools": [
                "get_portfolio_summary",
                "get_spending_drift",
            ],
            "notes": "Historical review limited to a subset.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "call_tool_payload",
        lambda tool_name, arguments=None, db_path=None: tool_calls.append((tool_name, arguments)) or {"ok": True},
    )
    monkeypatch.setattr(
        planner_agent,
        "generate_planner_response",
        lambda prompt_context: {
            "summary": "Subset historical review completed.",
            "highlights": [],
            "next_action": "No further action needed.",
        },
    )
    monkeypatch.setattr(
        planner_agent,
        "date",
        type(
            "FakeDate",
            (),
            {
                "today": staticmethod(lambda: __import__("datetime").date(2026, 4, 26)),
                "fromisoformat": staticmethod(__import__("datetime").date.fromisoformat),
            },
        ),
    )

    result = planner_agent.run_planner_agent("Review spending for last month")

    assert result["used_tools"] == [
        "get_portfolio_summary",
        "get_spending_drift",
    ]
    assert [tool_name for tool_name, _ in tool_calls] == [
        "get_portfolio_summary",
        "get_spending_drift",
    ]
