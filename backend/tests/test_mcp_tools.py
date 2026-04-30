import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.mcp.tools as mcp_tools


def test_register_tools_registers_budget_tool_names():
    registered_tools = []

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered_tools.append(kwargs.get("name"))
                return fn
            return decorator

    mcp_tools.register_tools(FakeFastMCP())

    assert registered_tools == [
        "health_check",
        "get_budget_status",
        "get_category_budget_status",
        "create_budget_plan",
        "update_budget_target",
        "get_portfolio_summary",
        "get_category_spend",
        "get_account_breakdown",
        "get_transaction_slice",
        "compare_periods",
        "get_spending_drift",
        "detect_spending_anomalies",
        "find_recurring_charges",
        "recommend_budget_targets",
    ]


def test_registered_health_check_returns_ok_payload():
    registered = {}

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    mcp_tools.register_tools(FakeFastMCP())

    payload = registered["health_check"]()
    assert payload == {
        "status": "ok",
        "server": "finance-planner",
    }


def test_registered_get_budget_status_calls_budget_service(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    expected = {"summary": {"total_target": 100.0}}
    monkeypatch.setattr(mcp_tools, "get_budget_status", lambda plan_id=None, db_path=None: expected)

    mcp_tools.register_tools(FakeFastMCP(), db_path="planner.sqlite")

    payload = registered["get_budget_status"]("plan-1")
    assert payload == expected


def test_registered_create_budget_plan_calls_budget_service(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    expected = {"plan_id": "plan-1"}
    monkeypatch.setattr(
        mcp_tools,
        "create_budget_plan",
        lambda period_start, period_end, targets, status="active", db_path=None: expected,
    )

    mcp_tools.register_tools(FakeFastMCP(), db_path="planner.sqlite")

    payload = registered["create_budget_plan"](
        "2026-04-01",
        "2026-04-30",
        [{"category_name": "Grocery", "target_amount": 500}],
        "active",
    )
    assert payload == expected


def test_registered_get_portfolio_summary_calls_ledger_service(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    expected = {"summary": {"total_expense": 100.0}}
    monkeypatch.setattr(
        mcp_tools,
        "get_portfolio_summary",
        lambda period_start, period_end, account_pid=None, account_name=None, db_path=None: expected,
    )

    mcp_tools.register_tools(FakeFastMCP(), db_path="planner.sqlite")

    payload = registered["get_portfolio_summary"]("2026-04-01", "2026-04-30")
    assert payload == expected


def test_registered_compare_periods_calls_ledger_service(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    expected = {"total_deltas": {"total_expense": 25.0}}
    monkeypatch.setattr(
        mcp_tools,
        "compare_periods",
        lambda current_start, current_end, previous_start, previous_end, db_path=None: expected,
    )

    mcp_tools.register_tools(FakeFastMCP(), db_path="planner.sqlite")

    payload = registered["compare_periods"]("2026-04-01", "2026-04-30", "2026-03-01", "2026-03-31")
    assert payload == expected


def test_registered_recommend_budget_targets_calls_recommendation_service(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    expected = {"planned_savings": 300.0}
    monkeypatch.setattr(
        mcp_tools,
        "recommend_budget_targets",
        lambda period_start, period_end, history_periods=3, savings_target=None, savings_rate=None, db_path=None: expected,
    )

    mcp_tools.register_tools(FakeFastMCP(), db_path="planner.sqlite")

    payload = registered["recommend_budget_targets"]("2026-04-28", "2026-05-27", 3, 300.0, None)
    assert payload == expected
