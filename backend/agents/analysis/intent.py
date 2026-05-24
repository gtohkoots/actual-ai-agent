from __future__ import annotations

from typing import Any, Optional

SUMMARY_KEYWORDS = {
    "overview",
    "overall",
    "summary",
    "summarize",
    "spending",
    "finances",
    "financial",
    "portfolio",
}
COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "versus",
    "vs",
    "changed",
    "change",
    "difference",
}
TREND_KEYWORDS = {
    "trend",
    "trending",
    "pattern",
    "patterns",
    "month",
    "monthly",
    "weekly",
}
ANOMALY_KEYWORDS = {
    "anomaly",
    "anomalies",
    "spike",
    "spikes",
    "unusual",
    "overspend",
    "overspending",
    "large",
}
CASHFLOW_KEYWORDS = {
    "cash flow",
    "cashflow",
    "income",
    "expense",
    "expenses",
    "net",
    "burn",
}
CATEGORY_KEYWORDS = {
    "category",
    "categories",
    "grocery",
    "groceries",
    "dining",
    "shopping",
    "gas",
    "bills",
    "rent",
}
PAYEE_KEYWORDS = {
    "merchant",
    "payee",
    "vendor",
    "costco",
    "amazon",
    "walmart",
    "target",
}
ACCOUNT_KEYWORDS = {
    "account",
    "card",
    "visa",
    "amex",
    "checking",
    "savings account",
}
PORTFOLIO_HINTS = {
    "my spending",
    "overall",
    "portfolio",
    "across all",
    "total spending",
    "financial picture",
    "my finances",
}


def _text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _contains_any(haystack: str, needles: set[str]) -> bool:
    return any(token in haystack for token in needles)


def interpret_analysis_request(request: Any) -> dict[str, Any]:
    message = _text(getattr(request, "message", ""))
    context = getattr(request, "context", None)
    focus_category = _text(getattr(context, "focus_category", None))
    focus_payee = _text(getattr(context, "focus_payee", None))
    account_name = _text(getattr(context, "account_name", None))
    card_label = _text(getattr(context, "card_label", None))
    account_pid = _text(getattr(context, "account_pid", None))

    intent = "summary"
    intent_reason = "Default to a general financial summary."
    if _contains_any(message, ANOMALY_KEYWORDS):
        intent = "anomaly_review"
        intent_reason = "The user asked about spikes, anomalies, or unusual spending."
    elif _contains_any(message, CASHFLOW_KEYWORDS):
        intent = "cashflow_review"
        intent_reason = "The user asked about income, expense, net flow, or burn."
    elif _contains_any(message, COMPARISON_KEYWORDS):
        intent = "comparison"
        intent_reason = "The user asked to compare periods or explain changes."
    elif _contains_any(message, TREND_KEYWORDS):
        intent = "trend_review"
        intent_reason = "The user asked about trends or patterns over time."
    elif focus_category or _contains_any(message, CATEGORY_KEYWORDS):
        intent = "category_deep_dive"
        intent_reason = "The user is focused on a category-level spending question."

    scope = "portfolio"
    scope_reason = "Default to the overall financial picture instead of a single card."
    entity: dict[str, Any] = {}
    if focus_payee or _contains_any(message, PAYEE_KEYWORDS):
        scope = "payee"
        scope_reason = "The user or UI context points to a merchant/payee-specific question."
        entity = {"payee": focus_payee or None}
    elif focus_category or _contains_any(message, CATEGORY_KEYWORDS):
        scope = "category"
        scope_reason = "The user or UI context points to a category-specific question."
        entity = {"category": focus_category or None}
    elif (account_pid or account_name or card_label) and _contains_any(message, ACCOUNT_KEYWORDS):
        scope = "account"
        scope_reason = "The user explicitly mentioned an account/card-oriented analysis."
        entity = {
            "account_pid": account_pid or None,
            "account_name": account_name or None,
            "card_label": card_label or None,
        }
    elif _contains_any(message, PORTFOLIO_HINTS):
        scope = "portfolio"
        scope_reason = "The user asked for an overall financial view."

    return {
        "intent": intent,
        "intent_reason": intent_reason,
        "scope": scope,
        "scope_reason": scope_reason,
        "entity": entity,
    }
