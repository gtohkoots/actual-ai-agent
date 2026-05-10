from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agents.prompts import PLANNER_AGENT_SYSTEM_PROMPT, build_planner_user_payload

TURN_INTENT_SYSTEM_PROMPT = """
You are an intent interpreter for a finance planner assistant.
Classify the user's latest message into one of:
- budget_review
- historical_review
- budget_recommendation
- budget_revision
- budget_approval

Return valid JSON only with keys:
- intent
- confidence
- needs_pending_recommendation
- allowed_tools
- notes

Rules:
- historical_review is for past-spending questions such as last month or previous month analysis.
- budget_recommendation is for creating or recommending a new budget draft.
- budget_revision is for changing an existing proposed budget draft and should only be used if a pending recommendation exists.
- budget_approval is for explicit approval to save a pending draft and should only be used if a pending recommendation exists.
- budget_review is the default when the user is asking about the current budget or when no other category cleanly fits.
- allowed_tools should contain only the tools that would be relevant for the chosen intent.
- Use exact tool names from this list when you populate allowed_tools:
  - budget_review: []
  - historical_review: get_portfolio_summary, get_category_spend, get_account_breakdown, get_spending_drift
  - budget_recommendation: recommend_budget_targets
  - budget_revision: revise_budget_recommendation
  - budget_approval: prepare_budget_plan_from_recommendation, create_budget_plan
""".strip()


BUDGET_REQUEST_SYSTEM_PROMPT = """
You are a structured interpreter for finance budget requests.
Extract budget request parameters from the user's message.

Return valid JSON only with keys:
- period_start
- period_end
- savings_target
- notes

Rules:
- period_start and period_end must be ISO dates in YYYY-MM-DD format.
- If the user gives an explicit date range, preserve it.
- If the user asks for "starting today for a month", use today through 29 days later.
- If the user asks for "next month", use the next full calendar month.
- savings_target must be a number when the user asks to save a specific amount, otherwise null.
- notes should briefly describe the interpretation.
""".strip()


INTENT_DEFAULT_TOOLS = {
    "budget_review": [],
    "historical_review": [
        "get_portfolio_summary",
        "get_category_spend",
        "get_account_breakdown",
        "get_spending_drift",
    ],
    "budget_recommendation": ["recommend_budget_targets"],
    "budget_revision": ["revise_budget_recommendation"],
    "budget_approval": [
        "prepare_budget_plan_from_recommendation",
        "create_budget_plan",
    ],
}

TOOL_NAME_ALIASES = {
    "recommend_budget": "recommend_budget_targets",
    "recommend_budget_target": "recommend_budget_targets",
    "recommend_budget_targets": "recommend_budget_targets",
    "revise_budget": "revise_budget_recommendation",
    "revise_budget_target": "revise_budget_recommendation",
    "revise_budget_targets": "revise_budget_recommendation",
    "revise_budget_recommendation": "revise_budget_recommendation",
    "prepare_budget_plan": "prepare_budget_plan_from_recommendation",
    "prepare_budget_from_recommendation": "prepare_budget_plan_from_recommendation",
    "prepare_budget_plan_from_recommendation": "prepare_budget_plan_from_recommendation",
    "create_budget": "create_budget_plan",
    "create_budget_plan": "create_budget_plan",
    "portfolio_summary": "get_portfolio_summary",
    "get_portfolio_summary": "get_portfolio_summary",
    "category_spend": "get_category_spend",
    "get_category_spend": "get_category_spend",
    "account_breakdown": "get_account_breakdown",
    "get_account_breakdown": "get_account_breakdown",
    "spending_drift": "get_spending_drift",
    "get_spending_drift": "get_spending_drift",
}


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


def interpret_planner_turn_intent(
    user_message: str,
    *,
    has_pending_recommendation: bool = False,
) -> dict[str, Any]:
    """Interpret the user's latest turn into structured intent for bounded routing."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _fallback_turn_intent(user_message, has_pending_recommendation=has_pending_recommendation)

    llm = ChatOpenAI(
        model=os.getenv("FINANCE_PLANNER_AGENT_MODEL", os.getenv("FINANCE_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=TURN_INTENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "user_message": user_message,
                            "has_pending_recommendation": has_pending_recommendation,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )
        parsed = _parse_model_payload(response.content)
        return _normalize_turn_intent(parsed, user_message, has_pending_recommendation=has_pending_recommendation)
    except Exception:
        return _fallback_turn_intent(user_message, has_pending_recommendation=has_pending_recommendation)


def interpret_budget_request_parameters(user_message: str) -> dict[str, Any]:
    """Interpret a budget-creation request into structured recommendation parameters."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _fallback_budget_request_parameters(user_message)

    llm = ChatOpenAI(
        model=os.getenv("FINANCE_PLANNER_AGENT_MODEL", os.getenv("FINANCE_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=BUDGET_REQUEST_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps({"user_message": user_message}, ensure_ascii=False, indent=2)),
            ]
        )
        parsed = _parse_model_payload(response.content)
        print(f"Raw model output for budget request parameters: {response.content}")
        return _normalize_budget_request_parameters(parsed, user_message)
    except Exception:
        return _fallback_budget_request_parameters(user_message)


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


def _normalize_turn_intent(
    payload: dict[str, Any],
    user_message: str,
    *,
    has_pending_recommendation: bool,
) -> dict[str, Any]:
    """Normalize model-produced intent payloads into a stable routing shape."""
    supported_intents = {
        "budget_review",
        "historical_review",
        "budget_recommendation",
        "budget_revision",
        "budget_approval",
    }
    intent = str(payload.get("intent", "")).strip()
    if intent not in supported_intents:
        return _fallback_turn_intent(user_message, has_pending_recommendation=has_pending_recommendation)

    normalized = {
        "intent": intent,
        "confidence": max(0.0, min(float(payload.get("confidence", 0.0)), 1.0)),
        "needs_pending_recommendation": bool(payload.get("needs_pending_recommendation", False)),
        "allowed_tools": _canonical_allowed_tools(intent, payload.get("allowed_tools", [])),
        "notes": str(payload.get("notes", "")).strip(),
    }
    if normalized["needs_pending_recommendation"] and not has_pending_recommendation:
        return _fallback_turn_intent(user_message, has_pending_recommendation=has_pending_recommendation)
    return normalized


def _normalize_budget_request_parameters(payload: dict[str, Any], user_message: str) -> dict[str, Any]:
    """Normalize model-produced budget request parameters into a stable shape."""
    period_start = str(payload.get("period_start", "")).strip()
    period_end = str(payload.get("period_end", "")).strip()
    if not _is_iso_date(period_start) or not _is_iso_date(period_end):
        return _fallback_budget_request_parameters(user_message)

    try:
        start = date.fromisoformat(period_start)
        end = date.fromisoformat(period_end)
    except ValueError:
        return _fallback_budget_request_parameters(user_message)

    if start > end:
        return _fallback_budget_request_parameters(user_message)

    savings_target = payload.get("savings_target")
    normalized_savings_target = None
    if savings_target is not None:
        try:
            normalized_savings_target = round(float(savings_target), 2)
        except (TypeError, ValueError):
            return _fallback_budget_request_parameters(user_message)
        if normalized_savings_target < 0:
            return _fallback_budget_request_parameters(user_message)

    return {
        "period_start": period_start,
        "period_end": period_end,
        "savings_target": normalized_savings_target,
        "notes": str(payload.get("notes", "")).strip(),
    }


def _is_iso_date(value: str) -> bool:
    """Return whether the string is a strict ISO date."""
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _fallback_budget_request_parameters(user_message: str) -> dict[str, Any]:
    """Provide deterministic parsing for common budget request date and savings phrases."""
    normalized = user_message.strip().lower()
    today = date.today()

    explicit_range = re.search(
        r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
        normalized,
    )
    if explicit_range:
        period_start = explicit_range.group(1)
        period_end = explicit_range.group(2)
    else:
        explicit_start = re.search(r"start(?:ing)?\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})", normalized)
        if explicit_start and "month" in normalized:
            start = date.fromisoformat(explicit_start.group(1))
            end = start + timedelta(days=29)
            period_start = start.isoformat()
            period_end = end.isoformat()
        elif "starting today" in normalized and "month" in normalized:
            period_start = today.isoformat()
            period_end = (today + timedelta(days=29)).isoformat()
        elif "next month" in normalized:
            next_month_start = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
            next_month_end = ((next_month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1))
            period_start = next_month_start.isoformat()
            period_end = next_month_end.isoformat()
        else:
            period_start = today.isoformat()
            period_end = (today + timedelta(days=29)).isoformat()

    savings_match = re.search(r"save\s+\$?(\d+(?:\.\d+)?)", normalized)
    savings_target = round(float(savings_match.group(1)), 2) if savings_match else None

    return {
        "period_start": period_start,
        "period_end": period_end,
        "savings_target": savings_target,
        "notes": "Deterministic fallback interpretation of the budget request.",
    }


def _canonical_allowed_tools(intent: str, raw_allowed_tools: Any) -> list[str]:
    """Normalize tool names to the supported MCP tool set for the chosen intent."""
    default_tools = list(INTENT_DEFAULT_TOOLS.get(intent, []))
    if not isinstance(raw_allowed_tools, list):
        return default_tools

    supported_tools = set(default_tools)
    normalized_tools: list[str] = []
    for item in raw_allowed_tools:
        tool_name = str(item).strip()
        if not tool_name:
            continue
        canonical = TOOL_NAME_ALIASES.get(tool_name, tool_name)
        if canonical in supported_tools and canonical not in normalized_tools:
            normalized_tools.append(canonical)

    return normalized_tools or default_tools


def _fallback_turn_intent(user_message: str, *, has_pending_recommendation: bool) -> dict[str, Any]:
    """Provide deterministic intent routing when model interpretation is unavailable."""
    normalized = user_message.strip().lower()

    approval_markers = [
        "approve this",
        "approve it",
        "save this budget",
        "save this plan",
        "create it",
        "looks good",
        "confirm this budget",
    ]
    if has_pending_recommendation and any(marker in normalized for marker in approval_markers):
        return {
            "intent": "budget_approval",
            "confidence": 0.9,
            "needs_pending_recommendation": True,
            "allowed_tools": list(INTENT_DEFAULT_TOOLS["budget_approval"]),
            "notes": "The user appears to be approving the currently pending draft.",
        }

    revision_markers = [
        "increase ",
        "decrease ",
        "reduce ",
        "raise ",
        "lower ",
        "don't touch",
        "do not touch",
        "keep savings",
        "instead",
    ]
    if has_pending_recommendation and any(marker in normalized for marker in revision_markers):
        return {
            "intent": "budget_revision",
            "confidence": 0.82,
            "needs_pending_recommendation": True,
            "allowed_tools": list(INTENT_DEFAULT_TOOLS["budget_revision"]),
            "notes": "The user appears to be changing the currently pending draft.",
        }

    budget_markers = [
        "create a budget",
        "recommend a budget",
        "budget starting",
        "budget for a month",
    ]
    if any(marker in normalized for marker in budget_markers):
        return {
            "intent": "budget_recommendation",
            "confidence": 0.88,
            "needs_pending_recommendation": False,
            "allowed_tools": list(INTENT_DEFAULT_TOOLS["budget_recommendation"]),
            "notes": "The user is asking for a new budget draft.",
        }

    historical_markers = [
        "last month",
        "previous month",
        "past month",
        "review spending",
        "spending behavior",
    ]
    if any(marker in normalized for marker in historical_markers):
        return {
            "intent": "historical_review",
            "confidence": 0.88,
            "needs_pending_recommendation": False,
            "allowed_tools": list(INTENT_DEFAULT_TOOLS["historical_review"]),
            "notes": "The user is asking for historical spending analysis.",
        }

    return {
        "intent": "budget_review",
        "confidence": 0.7,
        "needs_pending_recommendation": False,
        "allowed_tools": [],
        "notes": "Defaulted to current-budget review routing.",
    }


def _fallback_planner_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a deterministic response when the LLM is unavailable."""
    if prompt_context.get("review_mode") == "budget_approval":
        return _fallback_budget_approval_response(prompt_context)
    if prompt_context.get("review_mode") == "budget_revision":
        return _fallback_budget_revision_response(prompt_context)
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


def _fallback_budget_revision_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a deterministic explanation for a revised budget recommendation payload."""
    revised = prompt_context.get("tool_results", {}).get("revise_budget_recommendation", {})
    if not revised:
        return {
            "summary": "There is no pending budget draft to revise yet.",
            "highlights": ["Start by asking for a budget recommendation before requesting changes."],
            "next_action": "Ask for a new budget draft first, then revise it once a recommendation exists.",
        }

    period_start = revised.get("period_start", "the requested start date")
    period_end = revised.get("period_end", "the requested end date")
    planned_savings = revised.get("planned_savings", 0.0)
    revision_context = revised.get("revision_context", {})
    protected_categories = revision_context.get("protected_categories", [])
    category_targets = revised.get("category_targets", [])

    highlights: list[str] = []
    if protected_categories:
        highlights.append(f"Protected categories: {', '.join(protected_categories)}.")
    highlights.extend(
        [
            f'{item["category_name"]} is now set to ${item["recommended_target"]:.2f}.'
            for item in category_targets[:3]
            if "recommended_target" in item
        ]
    )
    if not highlights:
        highlights.append("The revised draft is ready for review.")

    return {
        "summary": (
            f"Revised budget draft for {period_start} to {period_end}: "
            f'planned savings remain ${planned_savings:.2f}.'
        ),
        "highlights": highlights,
        "next_action": "Review the revised draft and confirm whether you want to save it.",
    }


def _fallback_budget_approval_response(prompt_context: dict[str, Any]) -> dict[str, Any]:
    """Provide a deterministic confirmation after a budget draft is saved."""
    saved_plan = prompt_context.get("tool_results", {}).get("create_budget_plan", {})
    create_payload = prompt_context.get("tool_results", {}).get("prepare_budget_plan_from_recommendation", {})
    if not saved_plan:
        return {
            "summary": "There is no approved budget draft ready to save yet.",
            "highlights": ["A budget draft must exist before the planner can save it."],
            "next_action": "Ask for a budget recommendation first, then approve it when you are ready.",
        }

    period_start = saved_plan.get("period_start", create_payload.get("period_start", "the requested start date"))
    period_end = saved_plan.get("period_end", create_payload.get("period_end", "the requested end date"))
    targets = saved_plan.get("targets", create_payload.get("targets", []))
    savings_target = next(
        (
            item.get("target_amount", 0.0)
            for item in targets
            if item.get("category_name") == "Savings"
        ),
        0.0,
    )

    highlights = [
        f'{item["category_name"]} saved at ${item["target_amount"]:.2f}.'
        for item in targets[:3]
        if "target_amount" in item
    ]
    if savings_target:
        highlights.append(f"Savings was included as a real budget target at ${savings_target:.2f}.")
    if not highlights:
        highlights.append("The approved budget draft was saved successfully.")

    return {
        "summary": f"Saved budget plan for {period_start} to {period_end}.",
        "highlights": highlights,
        "next_action": "Review the new active plan and monitor it against live spending.",
    }
