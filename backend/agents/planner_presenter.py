from __future__ import annotations

from typing import Any


def render_planner_chat_content(turn_result: dict[str, Any]) -> str:
    """Render a frontend-friendly markdown reply from a planner turn result."""
    summary = _render_turn_summary(turn_result)
    highlights = _filtered_highlights(turn_result)
    next_action = str(turn_result.get("next_action", "")).strip()
    plan_details = _render_budget_plan_details(turn_result)
    overwrite_warning = _render_active_budget_overwrite_warning(turn_result)

    sections: list[str] = []
    if summary:
        sections.append(summary)
    if plan_details:
        sections.append(plan_details)
    if overwrite_warning:
        sections.append(overwrite_warning)
    if highlights:
        sections.append("**Highlights**\n" + "\n".join(f"- {item}" for item in highlights))
    if next_action:
        sections.append(f"**Next**\n{next_action}")
    return "\n\n".join(sections) if sections else "No planner response was generated."


def _render_turn_summary(turn_result: dict[str, Any]) -> str:
    """Prefer deterministic summaries for draft/save turns to avoid mixing active-plan context."""
    turn_intent = dict(turn_result.get("turn_intent", {}))
    intent = str(turn_intent.get("intent", "")).strip()
    tool_results = dict(turn_result.get("tool_results", {}))
    planner_state = dict(turn_result.get("updated_planner_state", turn_result.get("planner_state", {})) or {})

    if intent == "budget_recommendation":
        plan = dict(
            tool_results.get("recommend_budget_targets")
            or planner_state.get("pending_recommendation")
            or {}
        )
        return _render_recommendation_summary(plan, verb="drafted")

    if intent == "budget_revision":
        plan = dict(
            tool_results.get("revise_budget_recommendation")
            or planner_state.get("pending_recommendation")
            or {}
        )
        return _render_recommendation_summary(plan, verb="updated")

    return str(turn_result.get("summary", "")).strip()


def _render_recommendation_summary(plan: dict[str, Any], *, verb: str) -> str:
    if not plan:
        return ""

    period_start = str(plan.get("period_start", "")).strip()
    period_end = str(plan.get("period_end", "")).strip()
    planned_savings = plan.get("planned_savings")
    savings_text = (
        f" with planned savings of {_format_currency(planned_savings)}"
        if isinstance(planned_savings, (int, float))
        else ""
    )
    subject = "a budget" if verb == "drafted" else "the budget draft"
    if period_start and period_end:
        return f"I {verb} {subject} for {period_start} to {period_end}{savings_text}."
    return f"I {verb} {subject}{savings_text}."


def _filtered_highlights(turn_result: dict[str, Any]) -> list[str]:
    turn_intent = dict(turn_result.get("turn_intent", {}))
    intent = str(turn_intent.get("intent", "")).strip()
    highlights = [str(item).strip() for item in turn_result.get("highlights", []) if str(item).strip()]
    if intent not in {"budget_recommendation", "budget_revision"}:
        return highlights

    blocked_markers = [
        "current active budget",
        "current active budget plan",
        "current budget period",
        "current savings target",
        "active budget period",
    ]
    return [
        item for item in highlights
        if not any(marker in item.lower() for marker in blocked_markers)
    ]


def _render_active_budget_overwrite_warning(turn_result: dict[str, Any]) -> str:
    turn_intent = dict(turn_result.get("turn_intent", {}))
    intent = str(turn_intent.get("intent", "")).strip()
    if intent not in {"budget_recommendation", "budget_revision"}:
        return ""

    active_plan = dict(turn_result.get("active_budget_plan", {}) or {})
    if active_plan.get("status") == "missing" or not active_plan:
        return ""

    period_start = str(active_plan.get("period_start", "")).strip()
    period_end = str(active_plan.get("period_end", "")).strip()
    period_text = f" for {period_start} to {period_end}" if period_start and period_end else ""

    return (
        "**Heads up**\n"
        f"Approving this draft will replace your current active budget{period_text} "
        "and archive the existing plan."
    )


def _render_budget_plan_details(turn_result: dict[str, Any]) -> str:
    turn_intent = dict(turn_result.get("turn_intent", {}))
    intent = str(turn_intent.get("intent", "")).strip()
    tool_results = dict(turn_result.get("tool_results", {}))
    planner_state = dict(turn_result.get("updated_planner_state", turn_result.get("planner_state", {})) or {})

    if intent == "budget_recommendation":
        plan = dict(
            tool_results.get("recommend_budget_targets")
            or planner_state.get("pending_recommendation")
            or {}
        )
        return _render_recommendation_details(plan, heading="Proposed Budget")

    if intent == "budget_revision":
        plan = dict(
            tool_results.get("revise_budget_recommendation")
            or planner_state.get("pending_recommendation")
            or {}
        )
        return _render_recommendation_details(plan, heading="Updated Budget")

    if intent == "budget_approval":
        plan = dict(
            tool_results.get("create_budget_plan")
            or tool_results.get("prepare_budget_plan_from_recommendation")
            or planner_state.get("latest_saved_plan")
            or planner_state.get("last_create_payload")
            or {}
        )
        return _render_saved_plan_details(plan)

    return ""


def _render_recommendation_details(plan: dict[str, Any], *, heading: str) -> str:
    if not plan:
        return ""

    lines: list[str] = [f"**{heading}**"]
    period_start = str(plan.get("period_start", "")).strip()
    period_end = str(plan.get("period_end", "")).strip()
    if period_start and period_end:
        lines.append(f"- Period: {period_start} to {period_end}")

    planned_savings = plan.get("planned_savings")
    if isinstance(planned_savings, (int, float)):
        lines.append(f"- Planned savings: {_format_currency(planned_savings)}")

    total_budgeted_spend = plan.get("total_budgeted_spend")
    if isinstance(total_budgeted_spend, (int, float)):
        lines.append(f"- Total budgeted spend: {_format_currency(total_budgeted_spend)}")

    buffer_remaining = plan.get("buffer_remaining")
    if isinstance(buffer_remaining, (int, float)):
        lines.append(f"- Buffer remaining: {_format_currency(buffer_remaining)}")

    category_targets = [
        item for item in plan.get("category_targets", [])
        if isinstance(item, dict) and isinstance(item.get("recommended_target"), (int, float))
    ]
    if category_targets:
        lines.append("")
        lines.append("| Category | Target |")
        lines.append("| --- | ---: |")
        for item in category_targets:
            lines.append(
                f'| {item.get("category_name", "Uncategorized")} | '
                f'{_format_currency(item["recommended_target"])} |'
            )

    return "\n".join(lines)


def _render_saved_plan_details(plan: dict[str, Any]) -> str:
    if not plan:
        return ""

    lines: list[str] = ["**Saved Budget**"]
    period_start = str(plan.get("period_start", "")).strip()
    period_end = str(plan.get("period_end", "")).strip()
    if period_start and period_end:
        lines.append(f"- Period: {period_start} to {period_end}")

    targets = [
        item for item in plan.get("targets", [])
        if isinstance(item, dict) and isinstance(item.get("target_amount"), (int, float))
    ]
    if targets:
        lines.append("")
        lines.append("| Category | Target |")
        lines.append("| --- | ---: |")
        for item in targets:
            lines.append(
                f'| {item.get("category_name", "Uncategorized")} | '
                f'{_format_currency(item["target_amount"])} |'
            )

    return "\n".join(lines)


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"
