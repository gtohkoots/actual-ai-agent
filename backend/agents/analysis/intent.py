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
EXPLICIT_SCOPE_TYPES = {"portfolio", "account", "category", "payee"}


def _text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _contains_any(haystack: str, needles: set[str]) -> bool:
    return any(token in haystack for token in needles)


def _resolve_intent(message: str) -> tuple[str, str]:
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
    elif _contains_any(message, CATEGORY_KEYWORDS):
        intent = "category_deep_dive"
        intent_reason = "The user is focused on a category-level spending question."
    return intent, intent_reason


def _explicit_scope_scope(context: Any, account_pid: str, account_name: str, card_label: str) -> tuple[str, str, dict[str, Any]] | None:
    explicit_scope = _text(getattr(context, "scope_type", None))
    selected_account_pid = _text(getattr(context, "selected_account_pid", None))
    selected_account_name = _text(getattr(context, "selected_account_name", None))
    selected_category = _text(getattr(context, "selected_category", None))
    selected_payee = _text(getattr(context, "selected_payee", None))

    if explicit_scope not in EXPLICIT_SCOPE_TYPES:
        return None

    if explicit_scope == "category":
        if selected_category:
            return (
                "category",
                "The user explicitly selected a category scope in the UI.",
                {"category": selected_category},
            )
        return (
            "portfolio",
            "The UI selected category scope without a category, so the request was safely downgraded to portfolio scope.",
            {},
        )

    if explicit_scope == "payee":
        if selected_payee:
            return (
                "payee",
                "The user explicitly selected a payee scope in the UI.",
                {"payee": selected_payee},
            )
        return (
            "portfolio",
            "The UI selected payee scope without a payee, so the request was safely downgraded to portfolio scope.",
            {},
        )

    if explicit_scope == "account":
        if selected_account_pid or selected_account_name:
            entity = {
                "account_pid": selected_account_pid or None,
                "account_name": selected_account_name or None,
                "card_label": selected_account_name or card_label or None,
            }
            return (
                "account",
                "The user explicitly selected account scope in the UI.",
                entity,
            )
        return (
            "portfolio",
            "The UI selected account scope without an account, so the request was safely downgraded to portfolio scope.",
            {},
        )

    return (
        "portfolio",
        "The user explicitly selected portfolio scope in the UI.",
        {},
    )


def interpret_analysis_request(request: Any) -> dict[str, Any]:
    message = _text(getattr(request, "message", ""))
    context = getattr(request, "context", None)
    account_name = _text(getattr(context, "account_name", None))
    card_label = _text(getattr(context, "card_label", None))
    account_pid = _text(getattr(context, "account_pid", None))

    intent, intent_reason = _resolve_intent(message)

    explicit_scope_resolution = _explicit_scope_scope(context, account_pid, account_name, card_label)
    if explicit_scope_resolution is not None:
        scope, scope_reason, entity = explicit_scope_resolution
        return {
            "intent": intent,
            "intent_reason": intent_reason,
            "scope": scope,
            "scope_reason": scope_reason,
            "entity": entity,
        }

    scope = "portfolio"
    scope_reason = "Default to the overall financial picture instead of a single card."
    entity: dict[str, Any] = {}
    if _contains_any(message, PAYEE_KEYWORDS):
        scope = "payee"
        scope_reason = "The user asked a merchant/payee-specific question."
        entity = {"payee": None}
    elif _contains_any(message, CATEGORY_KEYWORDS):
        scope = "category"
        scope_reason = "The user asked a category-specific question."
        entity = {"category": None}
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
