from __future__ import annotations

from typing import Any, Optional

from backend.mcp.resources import read_active_budget_plan, read_current_budget_status


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

