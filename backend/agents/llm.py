from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.prompts import PLANNER_AGENT_SYSTEM_PROMPT, build_planner_user_payload


def generate_planner_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Use the configured chat model to synthesize a planner response from MCP context."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _fallback_planner_response(prompt_context)

    llm = ChatOpenAI(
        model=os.getenv("FINANCE_PLANNER_AGENT_MODEL", os.getenv("FINANCE_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=build_planner_user_payload(prompt_context)),
            ]
        )
        parsed = _parse_model_payload(response.content)
        return {
            "summary": str(parsed.get("summary", "")).strip() or _fallback_planner_response(prompt_context)["summary"],
            "highlights": [str(item) for item in parsed.get("highlights", []) if str(item).strip()],
            "next_action": str(parsed.get("next_action", "")).strip()
            or _fallback_planner_response(prompt_context)["next_action"],
        }
    except Exception:
        return _fallback_planner_response(prompt_context)


def _parse_model_payload(raw: Any) -> dict[str, Any]:
    """Parse JSON-only model output, tolerating fenced code blocks."""
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _fallback_planner_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a deterministic response when the LLM is unavailable."""
    if prompt_context.get("review_mode") == "budget_recommendation":
        return _fallback_budget_recommendation_response(prompt_context)
    if prompt_context.get("review_mode") == "historical_review":
        return _fallback_historical_response(prompt_context)

    active_plan = prompt_context.get("active_budget_plan", {})
    current_status = prompt_context.get("budget_status", {})

    if active_plan.get("status") == "missing" or current_status.get("status") == "missing":
        return {
            "summary": "No active budget plan is set up yet.",
            "highlights": [
                "The planner could not find an active budget plan to review.",
            ],
            "next_action": "Create a budget plan before requesting a budget review.",
        }

    summary = current_status["summary"]
    categories = current_status.get("categories", [])
    overspent = [item for item in categories if item.get("status") == "overspent"]
    at_risk = [item for item in categories if item.get("status") == "at_risk"]

    period = f'{current_status["period_start"]} to {current_status["period_end"]}'
    highlights: list[str] = []
    if overspent:
        highlights.extend(
            [
                f'{item["category_name"]} is overspent by ${abs(item["remaining_amount"]):.2f}.'
                for item in overspent
            ]
        )
    if at_risk:
        highlights.extend(
            [
                f'{item["category_name"]} is at risk with {item["utilization_pct"]:.1f}% of its target already used.'
                for item in at_risk
            ]
        )
    if not highlights:
        highlights.append("All budgeted categories are currently on track.")

    next_action = "No budget changes are needed right now; continue monitoring the current plan."
    if overspent:
        next_action = (
            "Review the overspent categories first and decide whether to reduce spending or raise their targets."
        )
    elif at_risk:
        next_action = (
            "Review the at-risk categories and decide whether to slow spending before the end of the period."
        )

    return {
        "summary": (
            f"Budget review for {period}: "
            f'target ${summary["total_target"]:.2f}, '
            f'actual ${summary["total_actual"]:.2f}, '
            f'remaining ${summary["total_remaining"]:.2f}.'
        ),
        "highlights": highlights,
        "next_action": next_action,
    }


def _fallback_historical_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a deterministic historical-spending review when the LLM is unavailable."""
    tool_results = prompt_context.get("tool_results", {})
    portfolio = tool_results.get("get_portfolio_summary", {})
    categories = tool_results.get("get_category_spend", {}).get("categories", [])
    accounts = tool_results.get("get_account_breakdown", {}).get("accounts", [])
    drift = tool_results.get("get_spending_drift", {}).get("top_category_changes", [])

    summary_data = portfolio.get("summary", {})
    period_start = portfolio.get("period_start", "the requested period")
    period_end = portfolio.get("period_end", "")
    period = period_start if not period_end else f"{period_start} to {period_end}"
    highlights: list[str] = []

    if categories:
        top_category = categories[0]
        highlights.append(
            f'{top_category["category_name"]} was the top spending category at ${top_category["amount"]:.2f}.'
        )
    if accounts:
        top_account = sorted(accounts, key=lambda item: item.get("expense_amount", 0.0), reverse=True)[0]
        highlights.append(
            f'{top_account["account_name"]} drove the most spending at ${top_account["expense_amount"]:.2f}.'
        )
    if drift:
        top_change = drift[0]
        direction = "up" if top_change.get("delta", 0.0) >= 0 else "down"
        highlights.append(
            f'{top_change["category_name"]} moved {direction} by ${abs(top_change["delta"]):.2f} versus the baseline period.'
        )
    if not highlights:
        highlights.append("No major historical spending driver was identified from the current context.")

    return {
        "summary": (
            f"Historical spending review for {period}: "
            f'total expense ${summary_data.get("total_expense", 0.0):.2f}, '
            f'total income ${summary_data.get("total_income", 0.0):.2f}, '
            f'net cash flow ${summary_data.get("net_cashflow", 0.0):.2f}.'
        ),
        "highlights": highlights,
        "next_action": "Review the top category changes and decide whether any of them should influence your next budget update.",
    }


def _fallback_budget_recommendation_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a deterministic explanation for a budget recommendation payload."""
    recommendation = prompt_context.get("tool_results", {}).get("recommend_budget_targets", {})
    period_start = recommendation.get("period_start", "the requested start date")
    period_end = recommendation.get("period_end", "the requested end date")
    planned_savings = recommendation.get("planned_savings", 0.0)
    total_budgeted_spend = recommendation.get("total_budgeted_spend", 0.0)
    category_targets = recommendation.get("category_targets", [])

    highlights = [
        (
            f'{item["category_name"]} is set to ${item["recommended_target"]:.2f} '
            f'from a baseline of ${item["baseline_amount"]:.2f}.'
        )
        for item in category_targets[:3]
        if "baseline_amount" in item and "recommended_target" in item
    ]
    if not highlights:
        highlights.append("No category targets were recommended from the available history.")

    return {
        "summary": (
            f"Recommended budget for {period_start} to {period_end}: "
            f'planned savings ${planned_savings:.2f}, '
            f'total budgeted spend ${total_budgeted_spend:.2f}.'
        ),
        "highlights": highlights,
        "next_action": "Review the draft and confirm whether you want to save it as a new budget plan.",
    }
