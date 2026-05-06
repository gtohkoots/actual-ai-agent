from __future__ import annotations

from typing import Any


def map_recommendation_to_budget_plan_payload(
    recommendation: dict[str, Any],
    *,
    status: str = "active",
    include_savings_target: bool = True,
) -> dict[str, Any]:
    """Convert a recommended budget payload into the create_budget_plan input shape."""
    period_start = str(recommendation["period_start"]).strip()
    period_end = str(recommendation["period_end"]).strip()
    if not period_start or not period_end:
        raise ValueError("Recommendation payload must include non-empty period_start and period_end values.")

    normalized_status = str(status).strip()
    if not normalized_status:
        raise ValueError("status cannot be empty.")

    seen_categories: set[str] = set()
    targets: list[dict[str, Any]] = []
    for item in recommendation.get("category_targets", []):
        category_name = str(item["category_name"]).strip()
        if not category_name:
            raise ValueError("Recommendation category_name cannot be empty.")
        if category_name in seen_categories:
            raise ValueError(f"Duplicate category target found in recommendation payload: {category_name}")
        recommended_target = round(float(item["recommended_target"]), 2)
        if recommended_target < 0:
            raise ValueError("Recommended target amount cannot be negative.")
        seen_categories.add(category_name)
        targets.append(
            {
                "category_name": category_name,
                "target_amount": recommended_target,
            }
        )

    planned_savings = round(float(recommendation.get("planned_savings", 0.0)), 2)
    if planned_savings < 0:
        raise ValueError("planned_savings cannot be negative.")

    if include_savings_target and planned_savings > 0:
        if "Savings" in seen_categories:
            targets = [
                {
                    "category_name": item["category_name"],
                    "target_amount": planned_savings if item["category_name"] == "Savings" else item["target_amount"],
                }
                for item in targets
            ]
        else:
            targets.append(
                {
                    "category_name": "Savings",
                    "target_amount": planned_savings,
                }
            )

    return {
        "period_start": period_start,
        "period_end": period_end,
        "targets": targets,
        "status": normalized_status,
    }
