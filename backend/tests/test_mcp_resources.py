import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.mcp.resources as resource_module
from backend.mcp.resources import read_active_budget_plan, read_current_budget_status


def test_read_active_budget_plan_returns_missing_payload_when_no_plan(monkeypatch):
    monkeypatch.setattr(resource_module, "get_active_budget_plan", lambda db_path=None: None)

    payload = read_active_budget_plan()

    assert payload == {
        "status": "missing",
        "message": "No active budget plan found.",
    }


def test_read_current_budget_status_returns_missing_payload_when_no_plan(monkeypatch):
    def raise_missing(db_path=None):
        raise KeyError("No active budget plan was found.")

    monkeypatch.setattr(resource_module, "get_budget_status", raise_missing)

    payload = read_current_budget_status()

    assert payload == {
        "status": "missing",
        "message": "No active budget plan found.",
    }


def test_read_current_budget_status_returns_live_status_payload(monkeypatch):
    expected = {
        "plan_id": "plan-1",
        "summary": {"total_target": 1000.0, "total_actual": 250.0, "total_remaining": 750.0, "utilization_pct": 25.0},
        "categories": [],
    }
    monkeypatch.setattr(resource_module, "get_budget_status", lambda db_path=None: expected)

    payload = read_current_budget_status()

    assert payload == expected
