import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app
from backend.services.planner_overview import generate_planner_overview


def test_generate_planner_overview_uses_mcp_backed_resource_readers(monkeypatch):
    expected_plan = {
        "plan_id": "plan-1",
        "period_start": "2026-05-01",
        "period_end": "2026-05-31",
        "status": "active",
        "targets": [{"category_name": "Bills", "target_amount": 1200.0}],
    }
    expected_status = {
        "plan_id": "plan-1",
        "summary": {
            "total_target": 2000.0,
            "total_actual": 620.0,
            "total_remaining": 1380.0,
            "utilization_pct": 31.0,
        },
        "categories": [],
    }

    monkeypatch.setattr(
        "backend.services.planner_overview.read_active_budget_plan",
        lambda db_path=None: expected_plan,
    )
    monkeypatch.setattr(
        "backend.services.planner_overview.read_current_budget_status",
        lambda db_path=None: expected_status,
    )

    payload = generate_planner_overview()

    assert payload.active_plan == expected_plan
    assert payload.current_status == expected_status


def test_planner_overview_endpoint_uses_overview_service(monkeypatch):
    expected = {
        "active_plan": {
            "plan_id": "plan-1",
            "period_start": "2026-05-01",
            "period_end": "2026-05-31",
            "status": "active",
            "targets": [{"category_name": "Bills", "target_amount": 1200.0}],
        },
        "current_status": {
            "plan_id": "plan-1",
            "summary": {
                "total_target": 2000.0,
                "total_actual": 620.0,
                "total_remaining": 1380.0,
                "utilization_pct": 31.0,
            },
            "categories": [],
        },
    }
    monkeypatch.setattr("backend.app.generate_planner_overview", lambda: expected)

    client = TestClient(app)
    response = client.get("/api/planner/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_plan"]["plan_id"] == "plan-1"
    assert payload["current_status"]["summary"]["total_actual"] == 620.0
