from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.services.budget_recommendations import recommend_budget_targets

REVISION_SYSTEM_PROMPT = """
You interpret user feedback on a proposed budget recommendation.
Use the current recommendation and the comment to return valid JSON only with keys:
- updated_savings_target
- protected_categories
- category_overrides
- notes

Rules:
- protected_categories must be an array of exact category names.
- category_overrides must be an object keyed by category name.
- Each override may contain amount_delta and target_amount.
- Prefer amount_delta when the user asks to raise/lower a category by implication.
- If the user says "a bit", infer a modest change based on the current recommendation.
- Do not invent categories that are not in the current recommendation.
- If no explicit savings change is requested, leave updated_savings_target as null.
""".strip()


def revise_budget_recommendation(
    current_recommendation: dict[str, Any],
    user_comment: str,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Interpret a comment about a recommendation and regenerate a new draft deterministically."""
    constraints = interpret_budget_revision_comment(current_recommendation, user_comment)
    revised = recommend_budget_targets(
        current_recommendation["period_start"],
        current_recommendation["period_end"],
        history_periods=int(current_recommendation.get("assumptions", {}).get("history_periods", 3)),
        savings_target=constraints.get("updated_savings_target") or current_recommendation.get("planned_savings"),
        category_overrides=constraints.get("category_overrides"),
        protected_categories=constraints.get("protected_categories"),
        db_path=db_path,
    )
    revised["revision_context"] = constraints
    revised["revision_comment"] = user_comment
    return revised


def interpret_budget_revision_comment(
    current_recommendation: dict[str, Any],
    user_comment: str,
) -> dict[str, Any]:
    """Convert a free-form revision comment into structured deterministic constraints."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _heuristic_revision_constraints(current_recommendation, user_comment)

    llm = ChatOpenAI(
        model=os.getenv("FINANCE_PLANNER_AGENT_MODEL", os.getenv("FINANCE_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
    )
    payload = {
        "current_recommendation": current_recommendation,
        "user_comment": user_comment,
    }
    try:
        response = llm.invoke(
            [
                SystemMessage(content=REVISION_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        parsed = _parse_json_payload(response.content)
        return _normalize_revision_constraints(current_recommendation, parsed, user_comment)
    except Exception:
        return _heuristic_revision_constraints(current_recommendation, user_comment)


def _heuristic_revision_constraints(
    current_recommendation: dict[str, Any],
    user_comment: str,
) -> dict[str, Any]:
    """Fallback parser for the most common budget revision phrases."""
    category_map = {
        item["category_name"].lower(): item
        for item in current_recommendation.get("category_targets", [])
        if item.get("category_name")
    }
    comment = user_comment.lower()
    protected_categories: list[str] = []
    category_overrides: dict[str, dict[str, Any]] = {}

    savings_match = re.search(
        r"(?:save|keep\s+savings\s+at|set\s+savings\s+to)\s+\$?(\d+(?:\.\d+)?)",
        comment,
    )
    updated_savings_target = round(float(savings_match.group(1)), 2) if savings_match else None

    for lower_name, item in category_map.items():
        proper_name = item["category_name"]
        if f"don't touch {lower_name}" in comment or f"do not touch {lower_name}" in comment or f"keep {lower_name} fixed" in comment:
            protected_categories.append(proper_name)
            continue

        current_target = float(item.get("recommended_target", 0.0))
        bit_delta = round(max(current_target * 0.1, 25.0), 2)

        if any(phrase in comment for phrase in [f"increase {lower_name}", f"raise {lower_name}", f"more {lower_name}"]):
            amount_match = re.search(rf"(?:increase|raise)\s+{re.escape(lower_name)}\s+(?:by\s+)?\$?(\d+(?:\.\d+)?)", comment)
            amount_delta = round(float(amount_match.group(1)), 2) if amount_match else bit_delta
            category_overrides[proper_name] = {"amount_delta": amount_delta}
            continue

        if any(phrase in comment for phrase in [f"reduce {lower_name}", f"lower {lower_name}", f"less {lower_name}", f"cut {lower_name}"]):
            amount_match = re.search(rf"(?:reduce|lower|cut)\s+{re.escape(lower_name)}\s+(?:by\s+)?\$?(\d+(?:\.\d+)?)", comment)
            amount_delta = round(float(amount_match.group(1)), 2) if amount_match else bit_delta
            category_overrides[proper_name] = {"amount_delta": -amount_delta}

    return {
        "updated_savings_target": updated_savings_target,
        "protected_categories": sorted(set(protected_categories)),
        "category_overrides": category_overrides,
        "notes": "Heuristic interpretation of the user's revision comment.",
    }


def _normalize_revision_constraints(
    current_recommendation: dict[str, Any],
    parsed: dict[str, Any],
    user_comment: str,
) -> dict[str, Any]:
    """Clean model-produced constraint payloads into the deterministic schema."""
    current_categories = {
        item["category_name"]
        for item in current_recommendation.get("category_targets", [])
        if item.get("category_name")
    }
    protected_categories = [
        category
        for category in parsed.get("protected_categories", [])
        if isinstance(category, str) and category in current_categories
    ]

    category_overrides: dict[str, dict[str, Any]] = {}
    raw_overrides = parsed.get("category_overrides", {})
    if isinstance(raw_overrides, dict):
        for category_name, override in raw_overrides.items():
            if category_name not in current_categories or not isinstance(override, dict):
                continue
            normalized_override: dict[str, Any] = {}
            if override.get("amount_delta") is not None:
                normalized_override["amount_delta"] = round(float(override["amount_delta"]), 2)
            if override.get("target_amount") is not None:
                normalized_override["target_amount"] = round(float(override["target_amount"]), 2)
            if normalized_override:
                category_overrides[category_name] = normalized_override

    updated_savings_target = parsed.get("updated_savings_target")
    return {
        "updated_savings_target": round(float(updated_savings_target), 2) if updated_savings_target is not None else None,
        "protected_categories": protected_categories,
        "category_overrides": category_overrides,
        "notes": str(parsed.get("notes", "")).strip() or f"Interpreted from user comment: {user_comment}",
    }


def _parse_json_payload(raw: Any) -> dict[str, Any]:
    """Parse JSON-only model output with light fence tolerance."""
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
