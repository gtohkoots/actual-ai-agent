from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd

from backend.utils.db import get_transactions_in_date_range

DEFAULT_SAVINGS_RATE = 0.10
WINDOW_WEIGHTS = [0.5, 0.3, 0.2]
MIN_CATEGORY_BASELINE = 10.0
MAX_CATEGORIES = 12

INFLOW_CATEGORIES = {
    "Rent Transfer",
    "income",
    "One-time deposit",
    "Paycheck",
    "Starting Balances",
}

RECURRING_INFLOW_CATEGORIES = {
    "Rent Transfer",
    "income",
    "Paycheck",
}

SAVINGS_CATEGORIES = {
    "Savings",
}

EXCLUDED_RECOMMENDATION_CATEGORIES = {
    "Ignored - expense",
    "Internal Transfer Expense",
    "Internal Transfer Income",
    "Lease Buyout",
    "One-time expense",
    "Pay Credit Card",
}

FIXED_EXPENSE_CATEGORIES = {
    "Bills",
    "Subscription",
}

ESSENTIAL_EXPENSE_CATEGORIES = {
    "Grocery",
    "Bills (Flexible)",
    "Gas",
    "Home",
    "Work",
}


def recommend_budget_targets(
    period_start: str,
    period_end: str,
    *,
    history_periods: int = 3,
    savings_target: Optional[float] = None,
    savings_rate: Optional[float] = None,
    category_overrides: Optional[dict[str, dict[str, Any]]] = None,
    protected_categories: Optional[list[str]] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Recommend category targets from recent ledger behavior while reserving savings first."""
    window_summaries = []
    for history_start, history_end in _build_history_windows(period_start, period_end, history_periods=history_periods):
        frame = _load_history_frame(history_start, history_end, db_path=db_path)
        window_summaries.append(
            {
                "period_start": history_start,
                "period_end": history_end,
                "income_total": _income_total(frame),
                "recurring_income_total": _recurring_income_total(frame),
                "savings_total": _savings_total(frame),
                "category_expense_totals": _category_expense_totals(frame),
            }
        )

    expected_income = _weighted_income(window_summaries)
    historical_savings_baseline = _weighted_average([item["savings_total"] for item in window_summaries])
    category_targets = _build_category_recommendations(window_summaries)
    chosen_savings_target, savings_target_source = _resolve_savings_target(
        expected_income,
        historical_savings_baseline=historical_savings_baseline,
        savings_target=savings_target,
        savings_rate=savings_rate,
    )
    category_targets = _apply_category_constraints(
        category_targets,
        category_overrides=category_overrides or {},
        protected_categories=protected_categories or [],
    )
    category_targets, total_budgeted_spend = _rebalance_for_savings(
        category_targets,
        expected_income=expected_income,
        savings_target=chosen_savings_target,
        protected_categories=protected_categories or [],
    )

    return {
        "period_start": period_start,
        "period_end": period_end,
        "history_windows": [
            {
                "period_start": item["period_start"],
                "period_end": item["period_end"],
                "income_total": item["income_total"],
            }
            for item in window_summaries
        ],
        "expected_income": expected_income,
        "planned_savings": chosen_savings_target,
        "historical_savings_baseline": historical_savings_baseline,
        "total_budgeted_spend": total_budgeted_spend,
        "buffer_remaining": round(expected_income - chosen_savings_target - total_budgeted_spend, 2),
        "category_targets": category_targets,
        "assumptions": {
            "history_periods": history_periods,
            "default_savings_rate": DEFAULT_SAVINGS_RATE,
            "applied_savings_rate": round(chosen_savings_target / expected_income, 4) if expected_income else 0.0,
            "savings_target_source": savings_target_source,
            "income_categories_used": sorted(INFLOW_CATEGORIES),
            "savings_categories_used": sorted(SAVINGS_CATEGORIES),
            "excluded_recommendation_categories": sorted(EXCLUDED_RECOMMENDATION_CATEGORIES),
            "protected_categories": protected_categories or [],
        },
    }


def _build_history_windows(period_start: str, period_end: str, *, history_periods: int) -> list[tuple[str, str]]:
    """Build trailing windows that match the requested future budget length."""
    start = date.fromisoformat(period_start)
    end = date.fromisoformat(period_end)
    window_days = (end - start).days + 1

    windows: list[tuple[str, str]] = []
    cursor_end = start - timedelta(days=1)
    for _ in range(history_periods):
        cursor_start = cursor_end - timedelta(days=window_days - 1)
        windows.append((cursor_start.isoformat(), cursor_end.isoformat()))
        cursor_end = cursor_start - timedelta(days=1)
    return windows


def _load_history_frame(period_start: str, period_end: str, *, db_path: Optional[str] = None) -> pd.DataFrame:
    """Load normalized history rows for recommendation analysis."""
    frame = get_transactions_in_date_range(
        period_start,
        period_end,
        db_path=db_path,
        join_names=True,
        dollars=True,
        debug=False,
    )
    if "category" not in frame.columns and "category_name" in frame.columns:
        frame["category"] = frame["category_name"]
    if "payee" not in frame.columns:
        frame["payee"] = None
    return frame.copy()


def _income_total(frame: pd.DataFrame) -> float:
    """Sum planner-relevant inflows based on the explicit category policy."""
    income_frame = frame.loc[frame["category"].isin(INFLOW_CATEGORIES)].copy()
    return round(float(income_frame.loc[income_frame["amount"] > 0, "amount"].sum()), 2)


def _recurring_income_total(frame: pd.DataFrame) -> float:
    """Sum the more stable inflow categories used for forward-looking income estimates."""
    recurring_income_frame = frame.loc[frame["category"].isin(RECURRING_INFLOW_CATEGORIES)].copy()
    return round(float(recurring_income_frame.loc[recurring_income_frame["amount"] > 0, "amount"].sum()), 2)


def _savings_total(frame: pd.DataFrame) -> float:
    """Sum money intentionally deposited to external savings as positive planning signal."""
    savings_frame = frame.loc[frame["category"].isin(SAVINGS_CATEGORIES)].copy()
    return round(float((-savings_frame.loc[savings_frame["amount"] < 0, "amount"]).sum()), 2)


def _category_expense_totals(frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate negative transactions into positive category expense totals."""
    if frame.empty:
        return {}
    expense_frame = frame.loc[
        (~frame["category"].isin(INFLOW_CATEGORIES))
        & (~frame["category"].isin(SAVINGS_CATEGORIES))
        & (~frame["category"].isin(EXCLUDED_RECOMMENDATION_CATEGORIES))
        & (frame["amount"] < 0)
    ].copy()
    grouped = (
        expense_frame.groupby("category", dropna=False)["amount"]
        .sum()
        .mul(-1.0)
    )
    return {
        (str(category) if category is not None else "(uncategorized)"): round(float(amount), 2)
        for category, amount in grouped.items()
    }


def _weighted_income(window_summaries: list[dict[str, Any]]) -> float:
    """Estimate the next-period income with more weight on recurring inflows."""
    recurring_income = _weighted_average([item["recurring_income_total"] for item in window_summaries])
    total_income = _weighted_average([item["income_total"] for item in window_summaries])
    if recurring_income > 0:
        supplemental_income = max(total_income - recurring_income, 0.0)
        return round(recurring_income + (supplemental_income * 0.35), 2)
    return total_income


def _build_category_recommendations(window_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert historical category behavior into initial target recommendations."""
    category_names = set()
    for item in window_summaries:
        category_names.update(item["category_expense_totals"].keys())

    recommendations: list[dict[str, Any]] = []
    for category_name in category_names:
        values = [item["category_expense_totals"].get(category_name, 0.0) for item in window_summaries]
        baseline_amount = _weighted_average(values)
        if baseline_amount < MIN_CATEGORY_BASELINE:
            continue
        category_type = _classify_category(category_name)
        recommended_target = _initial_target_for_category(category_type, baseline_amount, values)
        recommendations.append(
            {
                "category_name": category_name,
                "category_type": category_type,
                "baseline_amount": baseline_amount,
                "recommended_target": recommended_target,
                "adjustment_reason": _initial_reason(category_type),
            }
        )

    recommendations.sort(key=lambda item: item["baseline_amount"], reverse=True)
    return recommendations[:MAX_CATEGORIES]


def _classify_category(category_name: str) -> str:
    """Classify categories into fixed, essential, or discretionary planning buckets."""
    if category_name in FIXED_EXPENSE_CATEGORIES:
        return "fixed"
    if category_name in ESSENTIAL_EXPENSE_CATEGORIES:
        return "essential"
    return "discretionary"


def _initial_target_for_category(category_type: str, baseline_amount: float, values: list[float]) -> float:
    """Create the initial category target before income-fit rebalancing."""
    latest_amount = values[0] if values else baseline_amount
    if category_type == "fixed":
        return round(max(baseline_amount, latest_amount), 2)
    if category_type == "essential":
        return round(baseline_amount * 1.05, 2)
    return round(baseline_amount * 0.95, 2)


def _initial_reason(category_type: str) -> str:
    """Return an explanation for the category target starting point."""
    if category_type == "fixed":
        return "Held close to your recent recurring baseline."
    if category_type == "essential":
        return "Buffered slightly above your recent baseline to protect essential spending."
    return "Set slightly below your recent baseline to preserve room for savings."


def _resolve_savings_target(
    expected_income: float,
    *,
    historical_savings_baseline: float,
    savings_target: Optional[float],
    savings_rate: Optional[float],
) -> tuple[float, str]:
    """Choose the savings reservation used for the recommendation."""
    if savings_target is not None:
        return round(max(float(savings_target), 0.0), 2), "explicit_target"
    if savings_rate is not None:
        return round(max(float(savings_rate), 0.0) * expected_income, 2), "explicit_rate"
    if historical_savings_baseline > 0:
        return round(historical_savings_baseline, 2), "historical_savings"
    return round(DEFAULT_SAVINGS_RATE * expected_income, 2), "default_rate"


def _rebalance_for_savings(
    category_targets: list[dict[str, Any]],
    *,
    expected_income: float,
    savings_target: float,
    protected_categories: list[str],
) -> tuple[list[dict[str, Any]], float]:
    """Reduce flexible categories until spend plus savings fits the expected income."""
    categories = [dict(item) for item in category_targets]
    available_for_spend = max(round(expected_income - savings_target, 2), 0.0)
    total_budgeted_spend = round(sum(item["recommended_target"] for item in categories), 2)
    shortfall = round(total_budgeted_spend - available_for_spend, 2)

    if shortfall > 0:
        shortfall = _reduce_category_group(categories, "discretionary", shortfall, protected_categories=protected_categories)
    if shortfall > 0:
        shortfall = _reduce_category_group(categories, "essential", shortfall, protected_categories=protected_categories)

    total_budgeted_spend = round(sum(item["recommended_target"] for item in categories), 2)
    return categories, total_budgeted_spend


def _reduce_category_group(
    categories: list[dict[str, Any]],
    category_type: str,
    shortfall: float,
    *,
    protected_categories: list[str],
) -> float:
    """Reduce one category group proportionally and return any unresolved shortfall."""
    group = [
        item
        for item in categories
        if item["category_type"] == category_type
        and item["recommended_target"] > 0
        and item["category_name"] not in protected_categories
    ]
    group_total = sum(item["recommended_target"] for item in group)
    if not group or group_total <= 0:
        return shortfall

    reduction_ratio = min(shortfall / group_total, 1.0)
    total_reduced = 0.0
    for item in group:
        original = item["recommended_target"]
        reduction = round(original * reduction_ratio, 2)
        item["recommended_target"] = round(max(original - reduction, 0.0), 2)
        total_reduced += reduction
        if reduction > 0:
            item["adjustment_reason"] = (
                f'{item["adjustment_reason"]} Reduced to protect the savings target and keep the plan within expected income.'
            )

    return round(max(shortfall - total_reduced, 0.0), 2)


def _apply_category_constraints(
    category_targets: list[dict[str, Any]],
    *,
    category_overrides: dict[str, dict[str, Any]],
    protected_categories: list[str],
) -> list[dict[str, Any]]:
    """Apply explicit revision constraints before income-fit rebalancing."""
    updated_targets = [dict(item) for item in category_targets]
    for item in updated_targets:
        category_name = item["category_name"]
        if category_name in protected_categories:
            item["adjustment_reason"] = f'{item["adjustment_reason"]} Protected by user request.'

        override = category_overrides.get(category_name)
        if not override:
            continue
        if override.get("target_amount") is not None:
            item["recommended_target"] = round(max(float(override["target_amount"]), 0.0), 2)
            item["adjustment_reason"] = f'{item["adjustment_reason"]} Updated from explicit user target.'
            continue
        if override.get("amount_delta") is not None:
            item["recommended_target"] = round(max(item["recommended_target"] + float(override["amount_delta"]), 0.0), 2)
            item["adjustment_reason"] = f'{item["adjustment_reason"]} Adjusted based on user feedback.'
    return updated_targets


def _weighted_average(values: list[float]) -> float:
    """Compute a recency-weighted average for aligned history windows."""
    if not values:
        return 0.0
    weights = WINDOW_WEIGHTS[: len(values)]
    weight_total = sum(weights)
    normalized_weights = [weight / weight_total for weight in weights]
    return round(sum(value * weight for value, weight in zip(values, normalized_weights)), 2)
