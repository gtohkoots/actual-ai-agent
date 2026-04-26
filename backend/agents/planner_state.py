from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannerAgentState:
    """Carry the minimal state needed for one planner-agent review run."""

    user_message: str
    active_budget_plan: dict[str, Any]
    budget_status: dict[str, Any]
    prompt_context: dict[str, Any] = field(default_factory=dict)
    model_response: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    highlights: list[str] = field(default_factory=list)
    next_action: str = ""
    used_resources: list[str] = field(default_factory=list)
