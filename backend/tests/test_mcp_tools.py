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
