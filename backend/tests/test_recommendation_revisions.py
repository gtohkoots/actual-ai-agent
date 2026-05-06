import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.services.recommendation_revisions as revisions


def test_heuristic_revision_constraints_interprets_common_budget_feedback():
    current = {
        "category_targets": [
            {"category_name": "Bills", "recommended_target": 1200.0},
            {"category_name": "Dine", "recommended_target": 240.0},
            {"category_name": "Shopping", "recommended_target": 180.0},
        ]
    }

    result = revisions.interpret_budget_revision_comment(
        current,
        "Keep savings at $500, don't touch Bills, increase Dine a bit, and reduce Shopping instead.",
    )

    assert result["updated_savings_target"] == 500.0
    assert result["protected_categories"] == ["Bills"]
    assert result["category_overrides"]["Dine"]["amount_delta"] > 0
    assert result["category_overrides"]["Shopping"]["amount_delta"] < 0


def test_revise_budget_recommendation_reuses_deterministic_generator(monkeypatch):
    monkeypatch.setattr(
        revisions,
        "interpret_budget_revision_comment",
        lambda current_recommendation, user_comment: {
            "updated_savings_target": 500.0,
            "protected_categories": ["Bills"],
            "category_overrides": {"Dine": {"amount_delta": 25.0}},
            "notes": "Test constraints.",
        },
    )

    def fake_recommend(period_start, period_end, **kwargs):
        return {
            "period_start": period_start,
            "period_end": period_end,
            "planned_savings": kwargs["savings_target"],
            "category_targets": [
                {"category_name": "Bills", "recommended_target": 1200.0},
                {"category_name": "Dine", "recommended_target": 275.0},
            ],
        }

    monkeypatch.setattr(revisions, "recommend_budget_targets", fake_recommend)

    revised = revisions.revise_budget_recommendation(
        {
            "period_start": "2026-04-29",
            "period_end": "2026-05-28",
            "planned_savings": 500.0,
            "assumptions": {"history_periods": 3},
        },
        "increase Dine a bit",
    )

    assert revised["planned_savings"] == 500.0
    assert revised["revision_context"]["protected_categories"] == ["Bills"]
    assert revised["revision_comment"] == "increase Dine a bit"
