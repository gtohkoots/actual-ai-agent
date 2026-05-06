from __future__ import annotations

from typing import Any, Optional

from backend.mcp.resources import read_active_budget_plan, read_current_budget_status
from backend.services.budget_payloads import map_recommendation_to_budget_plan_payload
from backend.services.budget_recommendations import recommend_budget_targets
from backend.services.recommendation_revisions import revise_budget_recommendation


def register_prompts(mcp: Any, *, db_path: Optional[str] = None) -> None:
    """Register initial planner prompts on a FastMCP server."""

    @mcp.prompt(
        name="review_current_budget",
        description="Guide the assistant through a review of the active budget.",
    )
    def review_current_budget() -> str:
        active_plan = read_active_budget_plan(db_path=db_path)
        current_status = read_current_budget_status(db_path=db_path)
        return (
            "Review the user's current budget.\n\n"
            f"Active budget plan:\n{active_plan}\n\n"
            f"Current budget status:\n{current_status}\n\n"
            "Summarize the budget briefly, identify any categories that are at risk or overspent, "
            "and suggest the next best action without making persistent changes automatically."
        )

    @mcp.prompt(
        name="adjust_budget_target",
        description="Guide the assistant through adjusting a category target with approval.",
    )
    def adjust_budget_target(category_name: str, proposed_target_amount: float) -> str:
        active_plan = read_active_budget_plan(db_path=db_path)
        current_status = read_current_budget_status(db_path=db_path)
        return (
            "Help the user adjust a budget target.\n\n"
            f"Requested category: {category_name}\n"
            f"Proposed target amount: {proposed_target_amount}\n\n"
            f"Active budget plan:\n{active_plan}\n\n"
            f"Current budget status:\n{current_status}\n\n"
            "Explain the likely impact of changing this category target, mention any tradeoffs, "
            "and ask for confirmation before applying the update."
        )

    @mcp.prompt(
        name="recommend_budget_plan",
        description="Guide the assistant through proposing a new budget plan before saving it.",
    )
    def recommend_budget_plan(
        period_start: str,
        period_end: str,
        savings_target: float | None = None,
        savings_rate: float | None = None,
    ) -> str:
        recommended = recommend_budget_targets(
            period_start,
            period_end,
            savings_target=savings_target,
            savings_rate=savings_rate,
            db_path=db_path,
        )
        return (
            "Propose a new budget plan for the user.\n\n"
            f"Requested period: {period_start} to {period_end}\n"
            f"Recommended budget draft:\n{recommended}\n\n"
            "Explain how the category targets were derived from recent spending history, "
            "call out the savings assumption, and ask for confirmation before creating the plan."
        )

    @mcp.prompt(
        name="revise_budget_plan",
        description="Guide the assistant through revising a proposed budget plan from user feedback.",
    )
    def revise_budget_plan(current_recommendation: dict[str, Any], user_comment: str) -> str:
        revised = revise_budget_recommendation(
            current_recommendation=current_recommendation,
            user_comment=user_comment,
            db_path=db_path,
        )
        return (
            "Revise the user's proposed budget plan.\n\n"
            f"User feedback: {user_comment}\n\n"
            f"Revised recommendation:\n{revised}\n\n"
            "Explain what changed, which categories were protected or adjusted, "
            "and ask for confirmation before saving the revised plan."
        )

    @mcp.prompt(
        name="finalize_budget_plan",
        description="Guide the assistant through saving an approved budget recommendation or revised draft.",
    )
    def finalize_budget_plan(
        approved_recommendation: dict[str, Any],
        approval_note: str = "The user approved this budget plan.",
    ) -> str:
        create_payload = map_recommendation_to_budget_plan_payload(approved_recommendation)
        return (
            "Finalize the user's budget plan after explicit approval.\n\n"
            f"Approval note: {approval_note}\n\n"
            f"Approved recommendation:\n{approved_recommendation}\n\n"
            f"Mapped create_budget_plan payload:\n{create_payload}\n\n"
            "Use this flow whether the approved draft came directly from recommend_budget_targets "
            "or from revise_budget_recommendation.\n"
            "If the user has not explicitly approved the draft yet, stop and ask for confirmation first.\n"
            "After approval, do not call create_budget_plan with the recommendation payload directly. "
            "Always convert it into the create_budget_plan shape first, then pass the mapped payload to create_budget_plan."
        )
