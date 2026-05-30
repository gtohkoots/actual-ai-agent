from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from backend.services.filters import filter_internal_transfer_rows
from backend.utils.db import get_transactions_in_date_range

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_PAYEE_LIMIT = 50


def _default_window() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start.isoformat(), end.isoformat()


def list_analysis_categories(
    *,
    account_pid: Optional[str] = None,
    account_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[str]:
    window_start, window_end = start_date or _default_window()[0], end_date or _default_window()[1]
    frame = get_transactions_in_date_range(
        window_start,
        window_end,
        db_path=db_path,
        join_names=True,
        dollars=True,
        account_pid=account_pid,
        account_name=account_name,
        debug=False,
    )
    frame = filter_internal_transfer_rows(frame)
    expense_frame = frame.loc[frame["amount"] < 0].copy()
    category_column = "category_name" if "category_name" in expense_frame.columns else "category"
    if category_column not in expense_frame.columns or expense_frame.empty:
        return []

    categories = {
        str(value).strip()
        for value in expense_frame[category_column].dropna().tolist()
        if str(value).strip() and str(value).strip().lower() not in {"(uncategorized)", "unknown"}
    }
    return sorted(categories, key=str.lower)



def list_analysis_payees(
    *,
    account_pid: Optional[str] = None,
    account_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = DEFAULT_PAYEE_LIMIT,
    db_path: Optional[str] = None,
) -> list[str]:
    window_start, window_end = start_date or _default_window()[0], end_date or _default_window()[1]
    frame = get_transactions_in_date_range(
        window_start,
        window_end,
        db_path=db_path,
        join_names=True,
        dollars=True,
        account_pid=account_pid,
        account_name=account_name,
        debug=False,
    )
    frame = filter_internal_transfer_rows(frame)
    expense_frame = frame.loc[frame["amount"] < 0].copy()
    if "payee" not in expense_frame.columns or expense_frame.empty:
        return []

    payee_frame = expense_frame.copy()
    payee_frame["payee"] = payee_frame["payee"].fillna("").map(lambda value: str(value).strip())
    payee_frame = payee_frame.loc[payee_frame["payee"] != ""]
    payee_frame = payee_frame.loc[~payee_frame["payee"].str.lower().isin({"(unknown)", "unknown"})]
    if payee_frame.empty:
        return []

    grouped = (
        payee_frame.groupby("payee", dropna=False)
        .agg(
            transaction_count=("payee", "size"),
            total_spend=("amount", lambda values: round(float((-values).sum()), 2)),
            last_seen=("date", "max"),
        )
        .reset_index()
        .sort_values(["transaction_count", "total_spend", "last_seen", "payee"], ascending=[False, False, False, True])
    )
    return grouped["payee"].head(limit).tolist()
