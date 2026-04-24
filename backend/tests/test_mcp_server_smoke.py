import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.mcp.server as mcp_server
from backend.mcp.server import SERVER_NAME, SERVER_VERSION, build_runtime_server, create_server_definition, has_mcp_runtime
from backend.mcp.schemas import MCPServerDefinition


def test_create_server_definition_exposes_minimal_budget_surface():
    definition = create_server_definition()

    assert definition.name == SERVER_NAME
    assert definition.version == SERVER_VERSION
    assert [item.name for item in definition.resources] == [
        "planner://budget/active-plan",
        "planner://budget/current-status",
    ]
    assert [item.name for item in definition.tools] == [
        "health_check",
        "get_budget_status",
        "get_category_budget_status",
        "create_budget_plan",
        "update_budget_target",
    ]
    assert [item.name for item in definition.prompts] == [
        "review_current_budget",
        "adjust_budget_target",
    ]


def test_write_budget_tools_are_marked_as_approval_required():
    definition = create_server_definition()
    approval_map = {item.name: item.approval_required for item in definition.tools}

    assert approval_map["health_check"] is False
    assert approval_map["get_budget_status"] is False
    assert approval_map["get_category_budget_status"] is False
    assert approval_map["create_budget_plan"] is True
    assert approval_map["update_budget_target"] is True


def test_runtime_builder_returns_definition_even_without_runtime_dependency():
    runtime = build_runtime_server()

    if has_mcp_runtime():
        assert getattr(runtime, "name", None) == SERVER_NAME
    else:
        assert isinstance(runtime, MCPServerDefinition)
        assert runtime.name == SERVER_NAME
    assert isinstance(has_mcp_runtime(), bool)


def test_runtime_builder_registers_resources_when_fastmcp_is_available(monkeypatch):
    registered_resources = []
    registered_tools = []
    registered_prompts = []

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name

        def resource(self, uri, **kwargs):
            def decorator(fn):
                registered_resources.append((uri, kwargs.get("description")))
                return fn
            return decorator

        def tool(self, **kwargs):
            def decorator(fn):
                registered_tools.append((kwargs.get("name"), kwargs.get("description")))
                return fn
            return decorator

        def prompt(self, **kwargs):
            def decorator(fn):
                registered_prompts.append((kwargs.get("name"), kwargs.get("description")))
                return fn
            return decorator

    monkeypatch.setattr(mcp_server, "has_mcp_runtime", lambda: True)
    monkeypatch.setattr(mcp_server, "find_spec", lambda name: object())
    import sys as _sys
    fake_module = type("FakeModule", (), {"FastMCP": FakeFastMCP})()
    monkeypatch.setitem(_sys.modules, "fastmcp", fake_module)

    runtime = build_runtime_server()

    assert runtime.name == SERVER_NAME
    assert [item[0] for item in registered_resources] == [
        "planner://budget/active-plan",
        "planner://budget/current-status",
    ]
    assert [item[0] for item in registered_tools] == [
        "health_check",
        "get_budget_status",
        "get_category_budget_status",
        "create_budget_plan",
        "update_budget_target",
    ]
    assert [item[0] for item in registered_prompts] == [
        "review_current_budget",
        "adjust_budget_target",
    ]
