from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from backend.mcp.resources import read_active_budget_plan, read_current_budget_status


class PlannerOverviewResponse(BaseModel):
    active_plan: Dict[str, Any] = Field(default_factory=dict)
    current_status: Dict[str, Any] = Field(default_factory=dict)


def generate_planner_overview(*, db_path: Optional[str] = None) -> PlannerOverviewResponse:
    """Return the active budgeting plan and its live current-period status."""
    return PlannerOverviewResponse(
        active_plan=read_active_budget_plan(db_path=db_path),
        current_status=read_current_budget_status(db_path=db_path),
    )
