from __future__ import annotations

from typing import Any, Optional

from backend.services.budgets import (
    create_budget_plan,
    get_budget_status,
    get_category_budget_status,
    update_budget_target,
)


def register_tools(mcp: Any, *, db_path: Optional[str] = None) -> None:
    """Register the initial budget-focused planner tools on a FastMCP server."""

    @mcp.tool(
        name="health_check",
        description="Return a simple health payload confirming the planner MCP server is reachable.",
    )
    def mcp_health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "server": "finance-planner",
        }

    @mcp.tool(
        name="get_budget_status",
        description="Return budget target versus actual status for the active or requested plan.",
    )
    def mcp_get_budget_status(plan_id: str | None = None) -> dict[str, Any]:
        return get_budget_status(plan_id=plan_id, db_path=db_path)

    @mcp.tool(
        name="get_category_budget_status",
        description="Return target versus actual status for one budget category.",
    )
    def mcp_get_category_budget_status(category_name: str, plan_id: str | None = None) -> dict[str, Any]:
        return get_category_budget_status(category_name=category_name, plan_id=plan_id, db_path=db_path)

    @mcp.tool(
        name="create_budget_plan",
        description="Create a budget plan with category targets. Requires explicit user approval in the app layer.",
    )
    def mcp_create_budget_plan(
        period_start: str,
        period_end: str,
        targets: list[dict[str, Any]],
        status: str = "active",
    ) -> dict[str, Any]:
        return create_budget_plan(
            period_start=period_start,
            period_end=period_end,
            targets=targets,
            status=status,
            db_path=db_path,
        )

    @mcp.tool(
        name="update_budget_target",
        description="Insert or update one category target in an existing budget plan. Requires explicit user approval in the app layer.",
    )
    def mcp_update_budget_target(plan_id: str, category_name: str, target_amount: float) -> dict[str, Any]:
        return update_budget_target(
            plan_id=plan_id,
            category_name=category_name,
            target_amount=target_amount,
            db_path=db_path,
        )
