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
    cycle_income: float
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
    portfolio: Dict[str, Any] = Field(default_factory=dict)
    accounts: List[DashboardAccountSummary] = Field(default_factory=list)


def _today() -> date:
    return date.today()


def _dataset_window(db_path: Optional[str] = None) -> tuple[str, str]:
    today = _today()
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT MIN(date) AS first_date
            FROM transactions
            """
        ).fetchone()

    first_date = row["first_date"] if row else None
    if first_date is None:
        return today.isoformat(), today.isoformat()

    first_day = datetime.strptime(str(first_date), "%Y%m%d").date()
    return first_day.isoformat(), today.isoformat()


def _default_window() -> tuple[str, str]:
    return _dataset_window()


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


def _daily_series(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    series = (
        df.assign(
            day=pd.to_datetime(df["date"]).dt.date,
            income=df["amount"].where(df["amount"] > 0, 0.0),
            expense=df["amount"].where(df["amount"] < 0, 0.0).abs(),
        )
        .groupby("day", as_index=False)[["income", "expense"]]
        .sum()
        .sort_values("day")
    )
    return [
        {
            "date": row["day"].isoformat(),
            "income": round(float(row["income"]), 2),
            "expense": round(float(row["expense"]), 2),
            "net": round(float(row["income"] - row["expense"]), 2),
        }
        for _, row in series.iterrows()
    ]


def _top_categories_all_accounts(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    expense_df = df[df["amount"] < 0]
    if expense_df.empty:
        return []
    categories = (
        expense_df.groupby("category", dropna=False)["amount"].sum().mul(-1.0).sort_values(ascending=False).head(limit)
    )
    return [
        {"category": str(idx) if idx is not None else "(uncategorized)", "amount": round(float(val), 2)}
        for idx, val in categories.items()
    ]


def _top_payees_all_accounts(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    expense_df = df[df["amount"] < 0]
    if expense_df.empty:
        return []
    payees = (
        expense_df.groupby("payee", dropna=False)["amount"].sum().mul(-1.0).sort_values(ascending=False).head(limit)
    )
    return [
        {"payee": str(idx) if idx is not None else "(unknown)", "amount": round(float(val), 2)}
        for idx, val in payees.items()
    ]


def _category_mix(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    categories = _top_categories_all_accounts(df, limit=limit)
    total = sum(item["amount"] for item in categories) or 1.0
    return [
        {
            "category": item["category"],
            "amount": item["amount"],
            "share": round((item["amount"] / total) * 100.0, 1),
        }
        for item in categories
    ]


def _income_mix(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    income_df = df[df["amount"] > 0]
    if income_df.empty:
        return []
    sources = (
        income_df.groupby("payee", dropna=False)["amount"].sum().sort_values(ascending=False).head(limit)
    )
    total = float(sources.sum()) or 1.0
    return [
        {
            "source": str(idx) if idx is not None else "(unknown)",
            "amount": round(float(val), 2),
            "share": round((float(val) / total) * 100.0, 1),
        }
        for idx, val in sources.items()
    ]


def _top_movers(start_date: str, end_date: str, db_path: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    current_frame = get_transactions_in_date_range(
        start_date,
        end_date,
        db_path=db_path,
        join_names=True,
        dollars=True,
        debug=False,
    )
    prev_start, prev_end = _previous_window(start_date, end_date)
    previous_frame = get_transactions_in_date_range(
        prev_start,
        prev_end,
        db_path=db_path,
        join_names=True,
        dollars=True,
        debug=False,
    )
    current = (
        current_frame[current_frame["amount"] < 0]
        .groupby("category", dropna=False)["amount"]
        .sum()
        .mul(-1.0)
    )
    previous = (
        previous_frame[previous_frame["amount"] < 0]
        .groupby("category", dropna=False)["amount"]
        .sum()
        .mul(-1.0)
    )
    categories = list(set(current.index) | set(previous.index))
    movers: List[Dict[str, Any]] = []
    for category in categories:
        cur_value = float(current.get(category, 0.0))
        prev_value = float(previous.get(category, 0.0))
        movers.append(
            {
                "category": str(category) if category is not None else "(uncategorized)",
                "current": round(cur_value, 2),
                "previous": round(prev_value, 2),
                "delta": round(cur_value - prev_value, 2),
            }
        )
    movers.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return movers[:limit]


def _daily_heatmap(df: pd.DataFrame, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    spend_by_day = (
        df.assign(day=pd.to_datetime(df["date"]).dt.date)
        .groupby("day", as_index=False)["amount"]
        .sum()
    )
    spend_map = {row["day"]: round(abs(float(row["amount"])), 2) for _, row in spend_by_day.iterrows()}
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days: List[Dict[str, Any]] = []
    current = start
    week_index = 0
    while current <= end:
        if current.weekday() == 0 and current != start:
            week_index += 1
        days.append(
            {
                "date": current.isoformat(),
                "weekday": current.weekday(),
                "week": week_index,
                "amount": spend_map.get(current, 0.0),
            }
        )
        current += timedelta(days=1)
    return days


def _portfolio_overview(start_date: str, end_date: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    frame = get_transactions_in_date_range(
        start_date,
        end_date,
        db_path=db_path,
        join_names=True,
        dollars=True,
        debug=False,
    )
    rollups = get_week_rollups(start_date, end_date, df=frame)
    current_income = float(rollups["summary"]["total_income"])
    current_expense = float(rollups["summary"]["total_expense"])
    net_cashflow = float(rollups["summary"]["net_cashflow"])
    accounts = list_accounts(db_path=db_path)
    total_balance = round(sum(float(account.get("balance_current") or 0.0) for account in accounts), 2)
    return {
        "summary": {
            "totalBalance": _format_currency(total_balance),
            "totalIncome": _format_currency(current_income),
            "totalSpend": _format_currency(current_expense),
            "netCashFlow": _format_currency(net_cashflow),
        },
        "series": _daily_series(frame),
        "categoryMix": _category_mix(frame),
        "incomeMix": _income_mix(frame),
        "topCategories": _top_categories_all_accounts(frame),
        "topMerchants": _top_payees_all_accounts(frame),
        "topMovers": _top_movers(start_date, end_date, db_path=db_path),
        "dailyHeatmap": _daily_heatmap(frame, start_date, end_date),
    }


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
        "totalIncome": _format_currency(float(rollups["summary"]["total_income"])),
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
        cycle_income=float(rollups["summary"]["total_income"]),
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
    portfolio = _portfolio_overview(start_date, end_date, db_path=db_path)
    portfolio["accountComparison"] = [
        {
            "accountPid": item.account_pid,
            "accountName": item.account_name,
            "spend": item.cycle_spend,
            "income": item.cycle_income,
            "balance": item.balance_current,
        }
        for item in summaries
    ]
    return DashboardOverview(
        month_label=_month_label_for_window(start_date),
        selected_window="All time" if _default_window() == (start_date, end_date) else f"{start_date} to {end_date}",
        window=DashboardWindow(start=start_date, end=end_date),
        portfolio=portfolio,
        accounts=summaries,
    )
