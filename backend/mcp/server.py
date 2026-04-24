from __future__ import annotations

from typing import Any
from importlib.util import find_spec

from backend.mcp.prompts import register_prompts
from backend.mcp.resources import register_resources
from backend.mcp.schemas import MCPItemDefinition, MCPServerDefinition
from backend.mcp.tools import register_tools


SERVER_NAME = "finance-planner"
SERVER_VERSION = "0.1.0"


def create_server_definition() -> MCPServerDefinition:
    """Build the initial runtime-agnostic MCP surface for the planner."""
    return MCPServerDefinition(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        resources=(
            MCPItemDefinition(
                name="planner://budget/active-plan",
                item_type="resource",
                description="The currently active budget plan and its category targets.",
            ),
            MCPItemDefinition(
                name="planner://budget/current-status",
                item_type="resource",
                description="The active budget plan compared against live current-period spend.",
            ),
        ),
        tools=(
            MCPItemDefinition(
                name="health_check",
                item_type="tool",
                description="Return a simple health payload confirming the planner MCP server is reachable.",
            ),
            MCPItemDefinition(
                name="get_budget_status",
                item_type="tool",
                description="Return budget target versus actual status for the active or requested plan.",
            ),
            MCPItemDefinition(
                name="get_category_budget_status",
                item_type="tool",
                description="Return target versus actual status for one budget category.",
            ),
            MCPItemDefinition(
                name="create_budget_plan",
                item_type="tool",
                description="Create a budget plan with category targets.",
                approval_required=True,
            ),
            MCPItemDefinition(
                name="update_budget_target",
                item_type="tool",
                description="Insert or update one category target in an existing budget plan.",
                approval_required=True,
            ),
        ),
        prompts=(
            MCPItemDefinition(
                name="review_current_budget",
                item_type="prompt",
                description="Guide the assistant through a review of the active budget.",
            ),
            MCPItemDefinition(
                name="adjust_budget_target",
                item_type="prompt",
                description="Guide the assistant through adjusting a category target with approval.",
            ),
        ),
    )


def has_mcp_runtime() -> bool:
    """Return whether the FastMCP runtime library is available in the environment."""
    return find_spec("fastmcp") is not None


def build_runtime_server(*, db_path: str | None = None) -> Any:
    """Build a FastMCP runtime when available, else return the abstract definition."""
    if not has_mcp_runtime():
        return create_server_definition()

    from fastmcp import FastMCP

    mcp = FastMCP(name=SERVER_NAME)
    register_resources(mcp, db_path=db_path)
    register_tools(mcp, db_path=db_path)
    register_prompts(mcp, db_path=db_path)
    return mcp
