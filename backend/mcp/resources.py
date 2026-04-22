from __future__ import annotations

from typing import Any, Optional

from backend.services.budgets import get_active_budget_plan, get_budget_status


def read_active_budget_plan(*, db_path: Optional[str] = None) -> dict[str, Any]:
    """Return the currently active budget plan as an MCP resource payload."""
    plan = get_active_budget_plan(db_path=db_path)
    return plan or {"status": "missing", "message": "No active budget plan found."}


def read_current_budget_status(*, db_path: Optional[str] = None) -> dict[str, Any]:
    """Return active budget-versus-actual status as an MCP resource payload."""
    try:
        return get_budget_status(db_path=db_path)
    except KeyError:
        return {"status": "missing", "message": "No active budget plan found."}


def register_resources(mcp: Any, *, db_path: Optional[str] = None) -> None:
    """Register the initial planner budget resources on a FastMCP server."""

    @mcp.resource(
        "planner://budget/active-plan",
        name="planner://budget/active-plan",
        description="The currently active budget plan and its category targets.",
    )
    def active_budget_plan() -> dict[str, Any]:
        return read_active_budget_plan(db_path=db_path)

    @mcp.resource(
        "planner://budget/current-status",
        name="planner://budget/current-status",
        description="The active budget plan compared against live current-period spend.",
    )
    def current_budget_status() -> dict[str, Any]:
        return read_current_budget_status(db_path=db_path)

