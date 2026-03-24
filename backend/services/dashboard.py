from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from backend.services.insights import get_week_rollups
from backend.utils.db import get_connection, get_transactions_in_date_range


class AccountItem(BaseModel):
    account_pid: str
    account_name: str
    balance_current: float = 0.0
    transaction_count: int = 0
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None


class DashboardAccountSummary(BaseModel):
    account_pid: str
    account_name: str
    balance_current: float
    cycle_spend: float
    delta_percent: float
    delta_text: str
    utilization_text: str
    summary: Dict[str, Any]
    categories: List[Dict[str, Any]]
    merchants: List[Dict[str, Any]]
    transactions: List[Dict[str, Any]]
    quick_prompts: List[str]
    context: Dict[str, Any]


class DashboardWindow(BaseModel):
    start: str
    end: str


class DashboardOverview(BaseModel):
    month_label: str
    selected_window: str
    window: DashboardWindow
    accounts: List[DashboardAccountSummary] = Field(default_factory=list)


def _today() -> date:
    return date.today()


def _default_window() -> tuple[str, str]:
    today = _today()
    start = today.replace(day=1)
    return start.isoformat(), today.isoformat()


def _previous_window(start_date: str, end_date: str) -> tuple[str, str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    delta_days = (end - start).days + 1
    previous_end = start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=delta_days - 1)
    return previous_start.isoformat(), previous_end.isoformat()


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _cents_to_dollars(value: Optional[int]) -> float:
    if value is None:
        return 0.0
    return round(float(value) / 100.0, 2)


def _month_label_for_window(start_date: str) -> str:
    month_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    return f"{calendar.month_name[month_start.month]} {month_start.year}"


def list_accounts(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                a.id AS account_pid,
                a.name AS account_name,
                a.balance_current AS balance_current,
                COUNT(t.id) AS transaction_count,
                MIN(t.date) AS first_transaction_date,
                MAX(t.date) AS last_transaction_date
            FROM accounts a
            LEFT JOIN transactions t ON t.acct = a.id
            GROUP BY a.id, a.name
            ORDER BY a.name ASC
            """
        ).fetchall()

    accounts: List[Dict[str, Any]] = []
    for row in rows:
        first_date = row["first_transaction_date"]
        last_date = row["last_transaction_date"]
        accounts.append(
            {
                "account_pid": row["account_pid"],
                "account_name": row["account_name"],
                "balance_current": _cents_to_dollars(row["balance_current"]),
                "transaction_count": int(row["transaction_count"] or 0),
                "first_transaction_date": str(first_date) if first_date else None,
                "last_transaction_date": str(last_date) if last_date else None,
            }
        )
    return accounts


def _top_transactions(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows = df.sort_values(["date", "amount"], ascending=[False, True]).head(limit)
    return [
        {
            "date": pd.to_datetime(row["date"]).date().isoformat(),
            "merchant": row.get("payee") or "(unknown)",
            "category": row.get("category") or "(uncategorized)",
            "amount": round(float(row["amount"]), 2),
        }
        for _, row in rows.iterrows()
    ]


def _card_prompt(top_category: str, top_merchant: str, account_name: str) -> List[str]:
    return [
        f"Summarize why {account_name} is elevated this month",
        f"Which merchants are driving {top_category.lower()} spend?",
        f"Are there recurring charges I should review on {top_merchant}?",
    ]


def _build_summary_for_account(account: Dict[str, Any], start_date: str, end_date: str, db_path: Optional[str] = None) -> DashboardAccountSummary:
    account_pid = account["account_pid"]
    account_name = account["account_name"]
    balance_current = float(account.get("balance_current") or 0.0)

    current_df = get_transactions_in_date_range(
        start_date,
        end_date,
        db_path=db_path,
        join_names=True,
        dollars=True,
        account_pid=account_pid,
        account_name=account_name,
        debug=False,
    )
    prev_start, prev_end = _previous_window(start_date, end_date)
    previous_df = get_transactions_in_date_range(
        prev_start,
        prev_end,
        db_path=db_path,
        join_names=True,
        dollars=True,
        account_pid=account_pid,
        account_name=account_name,
        debug=False,
    )

    rollups = get_week_rollups(start_date, end_date, df=current_df)
    current_total = float(rollups["summary"]["total_expense"])
    previous_total = float(previous_df.loc[previous_df["amount"] < 0, "amount"].abs().sum().round(2))
    delta_percent = round(((current_total - previous_total) / previous_total) * 100, 2) if previous_total else (100.0 if current_total else 0.0)
    delta_text = f"{delta_percent:+.1f}% vs previous window"

    top_category = (rollups.get("by_category") or [{}])[0].get("category", "n/a")
    second_category = (rollups.get("by_category") or [{}, {}])[1].get("category") if len(rollups.get("by_category") or []) > 1 else None
    top_merchant = (rollups.get("top_payees") or [{}])[0].get("payee", "n/a")

    utilization_text = f"{top_category} heavy" if not second_category else f"{top_category} and {second_category} heavy"
    summary = {
        "totalSpend": _format_currency(current_total),
        "netCashFlow": _format_currency(float(rollups["summary"]["net_cashflow"])),
        "topCategory": top_category,
        "topMerchant": top_merchant,
        "aiSuggestion": f"Ask why {top_merchant} is the biggest driver on this account",
    }

    return DashboardAccountSummary(
        account_pid=account_pid,
        account_name=account_name,
        balance_current=balance_current,
        cycle_spend=current_total,
        delta_percent=delta_percent,
        delta_text=delta_text,
        utilization_text=utilization_text,
        summary=summary,
        categories=rollups.get("by_category", []),
        merchants=rollups.get("top_payees", []),
        transactions=_top_transactions(current_df, limit=5),
        quick_prompts=_card_prompt(top_category, top_merchant, account_name),
        context={
            "card": account_name,
            "accountName": account_name,
            "accountPid": account_pid,
            "dateRange": _month_label_for_window(start_date),
            "windowStart": start_date,
            "windowEnd": end_date,
            "focus": f"{top_category} + {top_merchant}",
        },
    )


def build_dashboard_overview(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> DashboardOverview:
    if not start_date or not end_date:
        start_date, end_date = _default_window()

    accounts = list_accounts(db_path=db_path)
    summaries = [
        _build_summary_for_account(account, start_date, end_date, db_path=db_path)
        for account in accounts
    ]
    summaries.sort(key=lambda item: item.cycle_spend, reverse=True)
    return DashboardOverview(
        month_label=_month_label_for_window(start_date),
        selected_window="Month to date" if _default_window() == (start_date, end_date) else f"{start_date} to {end_date}",
        window=DashboardWindow(start=start_date, end=end_date),
        accounts=summaries,
    )
