from __future__ import annotations

import json
from typing import Any

PLANNER_AGENT_SYSTEM_PROMPT = """
You are a personal finance planner assistant.
Use only the planner context provided to you.
Do not invent transactions, categories, forecasts, or recommendations that are not grounded in the context.
Do not apply any write actions or imply that a budget change has already been made.
Return valid JSON only with keys: summary, highlights, next_action.
summary must be a concise paragraph.
highlights must be an array of short concrete bullet-style strings.
next_action must be one sentence describing the best immediate follow-up.
If there is no active budget plan, explain that clearly and suggest creating one first.
""".strip()


def build_planner_prompt_context(
    user_message: str,
    active_budget_plan: dict[str, Any],
    budget_status: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the MCP-derived context the planner model is allowed to reason over."""
    return {
        "user_message": user_message,
        "active_budget_plan": active_budget_plan,
        "budget_status": budget_status,
    }


def build_planner_user_payload(prompt_context: dict[str, Any]) -> str:
    """Serialize planner context into a stable JSON payload for the model."""
    return json.dumps(prompt_context, ensure_ascii=False, indent=2)
