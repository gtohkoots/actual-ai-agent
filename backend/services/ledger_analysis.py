from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from backend.services.filters import filter_internal_transfer_rows
from backend.services.insights import detect_anomalies, find_recurring
from backend.utils.db import get_transactions_in_date_range


def get_portfolio_summary(
    period_start: str,
    period_end: str,
    *,
    account_pid: Optional[str] = None,
    account_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return a compact ledger-backed summary for one period across the selected scope."""
    frame = _load_period_frame(
        period_start,
        period_end,
        account_pid=account_pid,
        account_name=account_name,
        db_path=db_path,
    )
    summary = _summarize_frame(frame)
    return {
        "period_start": period_start,
        "period_end": period_end,
        "account_scope": _account_scope(account_pid=account_pid, account_name=account_name),
        "summary": summary,
    }


def get_category_spend(
    period_start: str,
    period_end: str,
    *,
    limit: int = 10,
    account_pid: Optional[str] = None,
    account_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return the top expense categories for a period with spend share percentages."""
    frame = _load_period_frame(
        period_start,
        period_end,
        account_pid=account_pid,
        account_name=account_name,
        db_path=db_path,
    )
    expense_frame = frame.loc[frame["amount"] < 0].copy()
    if expense_frame.empty:
        categories: list[dict[str, Any]] = []
        total_expense = 0.0
    else:
        grouped = (
            expense_frame.groupby("category", dropna=False)["amount"]
            .sum()
            .mul(-1.0)
            .sort_values(ascending=False)
            .head(limit)
        )
        total_expense = round(float(grouped.sum()), 2)
        categories = [
            {
                "category_name": str(category) if category is not None else "(uncategorized)",
                "amount": round(float(amount), 2),
                "share_pct": round((float(amount) / total_expense) * 100, 2) if total_expense else 0.0,
            }
            for category, amount in grouped.items()
        ]

    return {
        "period_start": period_start,
        "period_end": period_end,
        "account_scope": _account_scope(account_pid=account_pid, account_name=account_name),
        "total_expense": total_expense,
        "categories": categories,
    }


def get_account_breakdown(
    period_start: str,
    period_end: str,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return income, expense, and net cash flow grouped by account for a period."""
    frame = _load_period_frame(period_start, period_end, db_path=db_path)
    if frame.empty:
        accounts: list[dict[str, Any]] = []
    else:
        rows = []
        for account_name, group in frame.groupby("account", dropna=False):
            income_amount = round(float(group.loc[group["amount"] > 0, "amount"].sum()), 2)
            expense_amount = round(float((-group.loc[group["amount"] < 0, "amount"]).sum()), 2)
            rows.append(
                {
                    "account_name": str(account_name) if account_name is not None else "(unknown)",
                    "income_amount": income_amount,
                    "expense_amount": expense_amount,
                    "net_cashflow": round(income_amount - expense_amount, 2),
                }
            )
        accounts = sorted(rows, key=lambda item: item["expense_amount"], reverse=True)

    return {
        "period_start": period_start,
        "period_end": period_end,
        "accounts": accounts,
    }


def get_transaction_slice(
    period_start: str,
    period_end: str,
    *,
    category_name: Optional[str] = None,
    payee: Optional[str] = None,
    account_name: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return a bounded filtered transaction list for drill-down questions."""
    frame = _load_period_frame(period_start, period_end, db_path=db_path)

    if category_name:
        frame = frame.loc[frame["category"].fillna("").str.lower() == category_name.strip().lower()]
    if payee:
        frame = frame.loc[frame["payee"].fillna("").str.lower() == payee.strip().lower()]
    if account_name:
        frame = frame.loc[frame["account"].fillna("").str.lower() == account_name.strip().lower()]

    frame = frame.sort_values("date", ascending=False).head(limit)
    transactions = [
        {
            "date": pd.to_datetime(row["date"]).date().isoformat(),
            "payee": row.get("payee") or "(unknown)",
            "category_name": row.get("category") or "(uncategorized)",
            "account_name": row.get("account") or "(unknown)",
            "amount": round(float(row["amount"]), 2),
        }
        for _, row in frame.iterrows()
    ]

    return {
        "period_start": period_start,
        "period_end": period_end,
        "filters": {
            "category_name": category_name,
            "payee": payee,
            "account_name": account_name,
            "limit": limit,
        },
        "transactions": transactions,
    }


def compare_periods(
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compare two arbitrary periods using the same normalized ledger summary logic."""
    current_frame = _load_period_frame(current_start, current_end, db_path=db_path)
    previous_frame = _load_period_frame(previous_start, previous_end, db_path=db_path)
    current_summary = _summarize_frame(current_frame)
    previous_summary = _summarize_frame(previous_frame)

    total_deltas = {
        key: round(current_summary[key] - previous_summary[key], 2)
        for key in ["total_income", "total_expense", "net_cashflow"]
    }

    category_changes = _category_change_rows(current_frame, previous_frame)
    return {
        "current_period": {"start": current_start, "end": current_end, "summary": current_summary},
        "previous_period": {"start": previous_start, "end": previous_end, "summary": previous_summary},
        "total_deltas": total_deltas,
        "category_changes": category_changes,
    }


def get_spending_drift(
    period_start: str,
    period_end: str,
    *,
    baseline_start: Optional[str] = None,
    baseline_end: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compare a period against a baseline window to explain what changed in spending."""
    if baseline_start is None or baseline_end is None:
        baseline_start, baseline_end = _previous_matching_window(period_start, period_end)
    comparison = compare_periods(
        period_start,
        period_end,
        baseline_start,
        baseline_end,
        db_path=db_path,
    )
    top_changes = sorted(comparison["category_changes"], key=lambda item: abs(item["delta"]), reverse=True)[:5]
    return {
        "period_start": period_start,
        "period_end": period_end,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "total_deltas": comparison["total_deltas"],
        "top_category_changes": top_changes,
    }


def detect_spending_anomalies(
    period_start: str,
    period_end: str,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return expense outliers for a period using the existing anomaly detector."""
    frame = _load_period_frame(period_start, period_end, db_path=db_path)
    return {
        "period_start": period_start,
        "period_end": period_end,
        "anomalies": detect_anomalies(period_start, period_end, df=frame),
    }


def find_recurring_charges(
    period_start: str,
    period_end: str,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return recurring-charge candidates for a period using the existing recurring detector."""
    frame = _load_period_frame(period_start, period_end, db_path=db_path)
    return {
        "period_start": period_start,
        "period_end": period_end,
        "recurring_charges": find_recurring(period_start, period_end, df=frame),
    }


def _load_period_frame(
    period_start: str,
    period_end: str,
    *,
    account_pid: Optional[str] = None,
    account_name: Optional[str] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load normalized ledger rows for planner analysis, excluding internal transfers."""
    frame = get_transactions_in_date_range(
        period_start,
        period_end,
        db_path=db_path,
        join_names=True,
        dollars=True,
        account_pid=account_pid,
        account_name=account_name,
        debug=False,
    )
    if "category" not in frame.columns and "category_name" in frame.columns:
        frame["category"] = frame["category_name"]
    if "account" not in frame.columns and "account_name" in frame.columns:
        frame["account"] = frame["account_name"]
    if "payee" not in frame.columns:
        frame["payee"] = None
    return filter_internal_transfer_rows(frame)


def _summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Convert normalized ledger rows into compact period-level totals."""
    total_income = round(float(frame.loc[frame["amount"] > 0, "amount"].sum()), 2)
    total_expense = round(float((-frame.loc[frame["amount"] < 0, "amount"]).sum()), 2)
    return {
        "total_income": total_income,
        "total_expense": total_expense,
        "net_cashflow": round(total_income - total_expense, 2),
        "transaction_count": int(len(frame)),
    }


def _category_change_rows(current_frame: pd.DataFrame, previous_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build category delta rows between two periods for drift-style explanations."""
    current_series = (
        current_frame.loc[current_frame["amount"] < 0]
        .groupby("category", dropna=False)["amount"]
        .sum()
        .mul(-1.0)
    )
    previous_series = (
        previous_frame.loc[previous_frame["amount"] < 0]
        .groupby("category", dropna=False)["amount"]
        .sum()
        .mul(-1.0)
    )
    categories = set(current_series.index) | set(previous_series.index)
    rows = []
    for category in categories:
        current_amount = round(float(current_series.get(category, 0.0)), 2)
        previous_amount = round(float(previous_series.get(category, 0.0)), 2)
        rows.append(
            {
                "category_name": str(category) if category is not None else "(uncategorized)",
                "current_amount": current_amount,
                "previous_amount": previous_amount,
                "delta": round(current_amount - previous_amount, 2),
            }
        )
    return sorted(rows, key=lambda item: abs(item["delta"]), reverse=True)


def _previous_matching_window(period_start: str, period_end: str) -> tuple[str, str]:
    """Return the previous window with the same inclusive duration as the requested period."""
    start = datetime.strptime(period_start, "%Y-%m-%d").date()
    end = datetime.strptime(period_end, "%Y-%m-%d").date()
    window_days = (end - start).days + 1
    baseline_end = start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=window_days - 1)
    return baseline_start.isoformat(), baseline_end.isoformat()


def _account_scope(*, account_pid: Optional[str], account_name: Optional[str]) -> dict[str, Any]:
    """Return a small metadata object describing the applied account filter."""
    return {
        "account_pid": account_pid,
        "account_name": account_name,
    }
