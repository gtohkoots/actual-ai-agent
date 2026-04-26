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

    result = planner_agent.run_planner_agent("Review my budget")

    assert result["summary"] == "No active budget plan is set up yet."
    assert result["next_action"] == "Create a budget plan before requesting a budget review."
    assert result["used_resources"] == [
        "planner://budget/active-plan",
        "planner://budget/current-status",
    ]


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

    result = planner_agent.run_planner_agent("Review my budget")

    assert "Dining is already over target" in result["summary"]
    assert "Dining is overspent by $40.00." in result["highlights"]
    assert "Grocery is at risk with 90.0% of its target already used." in result["highlights"]
    assert result["next_action"] == (
        "Review the overspent categories first and decide whether to reduce spending or raise their targets."
    )
    assert result["prompt_context"]["budget_status"]["summary"]["total_actual"] == 620.0
