from __future__ import annotations

from typing import Any, Optional

from backend.services.budgets import (
    create_budget_plan,
    get_budget_status,
    get_category_budget_status,
    update_budget_target,
)
from backend.services.ledger_analysis import (
    compare_periods,
    detect_spending_anomalies,
    find_recurring_charges,
    get_account_breakdown,
    get_category_spend,
    get_portfolio_summary,
    get_spending_drift,
    get_transaction_slice,
)


def register_tools(mcp: Any, *, db_path: Optional[str] = None) -> None:
    """Register the budget and ledger-analysis planner tools on a FastMCP server."""

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

    @mcp.tool(
        name="get_portfolio_summary",
        description="Return income, expense, net cash flow, and transaction count for a requested period.",
    )
    def mcp_get_portfolio_summary(
        period_start: str,
        period_end: str,
        account_pid: str | None = None,
        account_name: str | None = None,
    ) -> dict[str, Any]:
        return get_portfolio_summary(
            period_start,
            period_end,
            account_pid=account_pid,
            account_name=account_name,
            db_path=db_path,
        )

    @mcp.tool(
        name="get_category_spend",
        description="Return the top expense categories for a requested period.",
    )
    def mcp_get_category_spend(
        period_start: str,
        period_end: str,
        limit: int = 10,
        account_pid: str | None = None,
        account_name: str | None = None,
    ) -> dict[str, Any]:
        return get_category_spend(
            period_start,
            period_end,
            limit=limit,
            account_pid=account_pid,
            account_name=account_name,
            db_path=db_path,
        )

    @mcp.tool(
        name="get_account_breakdown",
        description="Return income, expense, and net cash flow grouped by account for a requested period.",
    )
    def mcp_get_account_breakdown(period_start: str, period_end: str) -> dict[str, Any]:
        return get_account_breakdown(period_start, period_end, db_path=db_path)

    @mcp.tool(
        name="get_transaction_slice",
        description="Return a bounded filtered transaction list for a requested period.",
    )
    def mcp_get_transaction_slice(
        period_start: str,
        period_end: str,
        category_name: str | None = None,
        payee: str | None = None,
        account_name: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return get_transaction_slice(
            period_start,
            period_end,
            category_name=category_name,
            payee=payee,
            account_name=account_name,
            limit=limit,
            db_path=db_path,
        )

    @mcp.tool(
        name="compare_periods",
        description="Compare two periods and return total deltas plus category changes.",
    )
    def mcp_compare_periods(
        current_start: str,
        current_end: str,
        previous_start: str,
        previous_end: str,
    ) -> dict[str, Any]:
        return compare_periods(
            current_start,
            current_end,
            previous_start,
            previous_end,
            db_path=db_path,
        )

    @mcp.tool(
        name="get_spending_drift",
        description="Explain how spending changed against a baseline period.",
    )
    def mcp_get_spending_drift(
        period_start: str,
        period_end: str,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
    ) -> dict[str, Any]:
        return get_spending_drift(
            period_start,
            period_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            db_path=db_path,
        )

    @mcp.tool(
        name="detect_spending_anomalies",
        description="Return unusual expenses for a requested period.",
    )
    def mcp_detect_spending_anomalies(period_start: str, period_end: str) -> dict[str, Any]:
        return detect_spending_anomalies(period_start, period_end, db_path=db_path)

    @mcp.tool(
        name="find_recurring_charges",
        description="Return recurring-charge candidates for a requested period.",
    )
    def mcp_find_recurring_charges(period_start: str, period_end: str) -> dict[str, Any]:
        return find_recurring_charges(period_start, period_end, db_path=db_path)
