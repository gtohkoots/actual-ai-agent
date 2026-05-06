import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.budget_payloads import map_recommendation_to_budget_plan_payload


def test_map_recommendation_to_budget_plan_payload_includes_savings_by_default():
    recommendation = {
        "period_start": "2026-05-03",
        "period_end": "2026-06-01",
        "planned_savings": 500.0,
        "category_targets": [
            {"category_name": "Bills", "recommended_target": 1200.0},
            {"category_name": "Grocery", "recommended_target": 483.0},
            {"category_name": "Dine", "recommended_target": 272.0},
        ],
    }

    payload = map_recommendation_to_budget_plan_payload(recommendation)

    assert payload == {
        "period_start": "2026-05-03",
        "period_end": "2026-06-01",
        "targets": [
            {"category_name": "Bills", "target_amount": 1200.0},
            {"category_name": "Grocery", "target_amount": 483.0},
            {"category_name": "Dine", "target_amount": 272.0},
            {"category_name": "Savings", "target_amount": 500.0},
        ],
        "status": "active",
    }


def test_map_recommendation_to_budget_plan_payload_can_exclude_savings_target():
    recommendation = {
        "period_start": "2026-05-03",
        "period_end": "2026-06-01",
        "planned_savings": 500.0,
        "category_targets": [
            {"category_name": "Bills", "recommended_target": 1200.0},
            {"category_name": "Grocery", "recommended_target": 483.0},
        ],
    }

    payload = map_recommendation_to_budget_plan_payload(
        recommendation,
        include_savings_target=False,
        status="draft",
    )

    assert payload == {
        "period_start": "2026-05-03",
        "period_end": "2026-06-01",
        "targets": [
            {"category_name": "Bills", "target_amount": 1200.0},
            {"category_name": "Grocery", "target_amount": 483.0},
        ],
        "status": "draft",
    }


def test_map_recommendation_to_budget_plan_payload_rejects_duplicate_categories():
    recommendation = {
        "period_start": "2026-05-03",
        "period_end": "2026-06-01",
        "planned_savings": 500.0,
        "category_targets": [
            {"category_name": "Bills", "recommended_target": 1200.0},
            {"category_name": "Bills", "recommended_target": 1100.0},
        ],
    }

    with pytest.raises(ValueError, match="Duplicate category target"):
        map_recommendation_to_budget_plan_payload(recommendation)
