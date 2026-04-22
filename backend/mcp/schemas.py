from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MCPItemType = Literal["resource", "tool", "prompt"]


@dataclass(frozen=True)
class MCPItemDefinition:
    """Describe one MCP-exposed capability before wiring a concrete runtime."""

    name: str
    item_type: MCPItemType
    description: str
    approval_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPServerDefinition:
    """Describe the planner MCP surface in a runtime-agnostic way."""

    name: str
    version: str
    resources: tuple[MCPItemDefinition, ...] = ()
    tools: tuple[MCPItemDefinition, ...] = ()
    prompts: tuple[MCPItemDefinition, ...] = ()

