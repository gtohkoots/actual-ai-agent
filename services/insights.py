# services/insights.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    # Prefer your existing loader if present
    from utils.db import get_transactions_in_date_range
except Exception:
    get_transactions_in_date_range = None  # type: ignore


# -------------------------
# Helpers
# -------------------------

def _ensure_df(df: Optional[pd.DataFrame], start_date: str, end_date: str) -> pd.DataFrame:
    if df is not None:
        out = df.copy()
    else:
        if get_transactions_in_date_range is None:
            raise RuntimeError("get_transactions_in_date_range not available. Import utils.db or pass df explicitly.")
        out = get_transactions_in_date_range(start_date, end_date)
    # Ensure types
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"])  # keep datetime64[ns]
    # Ensure dollars (float)
    if pd.api.types.is_integer_dtype(out["amount"]) or (out["amount"].abs().max() > 10000):
        out["amount"] = (out["amount"].astype(float) / 100).round(2)
    else:
        out["amount"] = out["amount"].astype(float).round(2)
    # Minimal expected columns
    for col in ["payee", "category"]:
        if col not in out.columns:
            out[col] = None
    return out.sort_values("date").reset_index(drop=True)


def _pos(x: float) -> float:
    return float(x) if x > 0 else 0.0


def _neg_abs(x: float) -> float:
    return float(-x) if x < 0 else 0.0


# -------------------------
# Public: Weekly rollups & insights
# -------------------------

def get_week_rollups(
    start_date: str,
    end_date: str,
    df: Optional[pd.DataFrame] = None,
    top_n_categories: int = 5,
    top_n_payees: int = 5,
    big_expense_threshold: float = 200.0,
) -> Dict:
    """
    Return weekly rollup metrics and shortlists, JSON-serializable.
    All amounts are in USD (positive numbers for totals).
    """
    data = _ensure_df(df, start_date, end_date)

    income_df = data[data["amount"] > 0]
    expense_df = data[data["amount"] < 0]

    total_income = float(income_df["amount"].sum().round(2))
    total_expense_abs = float((-expense_df["amount"]).sum().round(2))
    net_cashflow = round(total_income - total_expense_abs, 2)

    # by category (expenses only, positive values)
    cat = (
        expense_df
        .groupby("category", dropna=False)["amount"].sum()
        .mul(-1.0)
        .sort_values(ascending=False)
        .head(top_n_categories)
    )
    by_category = [
        {"category": str(idx) if idx is not None else "(uncategorized)", "amount": round(val, 2)}
        for idx, val in cat.items()
    ]

    # top payees by absolute outgoing
    pay = (
        expense_df
        .groupby("payee", dropna=False)["amount"].sum()
        .mul(-1.0)
        .sort_values(ascending=False)
        .head(top_n_payees)
    )
    top_payees = [
        {"payee": str(idx) if idx is not None else "(unknown)", "amount": round(val, 2)}
        for idx, val in pay.items()
    ]

    # big expenses list
    big_expenses_df = expense_df.copy()
    big_expenses_df["abs_amount"] = big_expenses_df["amount"].abs()
    big_expenses_df = big_expenses_df[big_expenses_df["abs_amount"] >= big_expense_threshold]
    big_expenses = [
        {
            "date": pd.to_datetime(row["date"]).date().isoformat(),
            "payee": row.get("payee") or "(unknown)",
            "category": row.get("category") or "(uncategorized)",
            "amount": round(abs(float(row["amount"])), 2),
        }
        for _, row in big_expenses_df.sort_values("abs_amount", ascending=False).iterrows()
    ]

    return {
        "window": {"start": start_date, "end": end_date},
        "summary": {
            "total_income": round(total_income, 2),
            "total_expense": round(total_expense_abs, 2),
            "net_cashflow": round(net_cashflow, 2),
        },
        "by_category": by_category,
        "top_payees": top_payees,
        "big_expenses": big_expenses,
    }


def compare_week_over_week(
    start_date: str,
    end_date: str,
    df: Optional[pd.DataFrame] = None,
    top_n_category_changes: int = 10,
) -> Dict:
    """Compare current window with previous week. Return deltas and category change leaderboard."""
    this_df = _ensure_df(df, start_date, end_date)

    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    last_start = (s - timedelta(days=7)).isoformat()
    last_end = (e - timedelta(days=7)).isoformat()

    prev_df = _ensure_df(None, last_start, last_end)

    def totals(frame: pd.DataFrame) -> Dict[str, float]:
        inc = float(frame.loc[frame["amount"] > 0, "amount"].sum().round(2))
        exp = float((-frame.loc[frame["amount"] < 0, "amount"]).sum().round(2))
        return {"income": inc, "expense": exp, "net": round(inc - exp, 2)}

    cur = totals(this_df)
    prev = totals(prev_df)
    delta = {
        k: round(cur[k] - prev[k], 2) for k in ["income", "expense", "net"]
    }
    pct = {
        k: (round((delta[k] / prev[k]) * 100, 2) if prev[k] else (100.0 if cur[k] else 0.0))
        for k in ["income", "expense", "net"]
    }

    # Category changes on expenses
    def exp_by_cat(frame: pd.DataFrame) -> pd.Series:
        return (
            frame[frame["amount"] < 0]
            .groupby("category", dropna=False)["amount"].sum()
            .mul(-1.0)
        )

    cur_cat = exp_by_cat(this_df)
    prev_cat = exp_by_cat(prev_df)
    cats = list(set(cur_cat.index) | set(prev_cat.index))
    changes = []
    for c in cats:
        cur_v = float(cur_cat.get(c, 0.0))
        prev_v = float(prev_cat.get(c, 0.0))
        d = round(cur_v - prev_v, 2)
        p = round((d / prev_v) * 100, 2) if prev_v else (100.0 if cur_v else 0.0)
        changes.append({"category": str(c), "this_week": cur_v, "last_week": prev_v, "delta": d, "pct_change": p})

    changes = sorted(changes, key=lambda x: abs(x["delta"]), reverse=True)[:top_n_category_changes]

    return {
        "window": {"start": start_date, "end": end_date},
        "previous_window": {"start": last_start, "end": last_end},
        "totals": {"this_week": cur, "last_week": prev, "delta": delta, "pct_change": pct},
        "category_changes": changes,
    }


def find_recurring(
    start_date: str,
    end_date: str,
    df: Optional[pd.DataFrame] = None,
    min_occurrences: int = 3,
    tolerance_days: int = 3,
) -> List[Dict]:
    """
    Naive recurring detector by payee: looks for repeated transactions with roughly regular intervals.
    Returns a list of candidates with avg_period_days and stats.
    """
    data = _ensure_df(df, start_date, end_date)

    rec = []
    for payee, g in data.groupby("payee", dropna=False):
        g = g.sort_values("date")
        if len(g) < min_occurrences:
            continue
        # day gaps
        days = g["date"].diff().dt.days.dropna()
        if len(days) < 2:
            continue
        avg = float(days.mean())
        std = float(days.std(ddof=0)) if len(days) > 1 else 0.0
        # amount stability (coefficient of variation on abs amounts)
        abs_amt = g["amount"].abs()
        if abs_amt.mean() == 0:
            cv = 0.0
        else:
            cv = float(abs_amt.std(ddof=0) / abs_amt.mean())
        if avg > 0 and std <= tolerance_days:
            rec.append({
                "payee": str(payee) if payee is not None else "(unknown)",
                "occurrences": int(len(g)),
                "avg_period_days": round(avg, 1),
                "period_std_days": round(std, 1),
                "amount_avg": round(float(abs_amt.mean()), 2),
                "amount_cv": round(cv, 2),
                "first_date": pd.to_datetime(g["date"].iloc[0]).date().isoformat(),
                "last_date": pd.to_datetime(g["date"].iloc[-1]).date().isoformat(),
            })
    return sorted(rec, key=lambda x: (-x["occurrences"], x["avg_period_days"]))


def detect_anomalies(
    start_date: str,
    end_date: str,
    df: Optional[pd.DataFrame] = None,
    method: str = "zscore",
    z: float = 2.5,
) -> List[Dict]:
    """Simple amount outlier detector on expenses (absolute size)."""
    data = _ensure_df(df, start_date, end_date)
    exp = data[data["amount"] < 0].copy()
    exp["abs_amount"] = exp["amount"].abs()
    if exp.empty:
        return []

    if method == "zscore":
        mu = exp["abs_amount"].mean()
        sd = exp["abs_amount"].std(ddof=0) or 1.0
        exp["z"] = (exp["abs_amount"] - mu) / sd
        out = exp[exp["z"] >= z].sort_values("z", ascending=False)
    else:  # IQR
        q1 = exp["abs_amount"].quantile(0.25)
        q3 = exp["abs_amount"].quantile(0.75)
        iqr = q3 - q1
        thr = q3 + 1.5 * iqr
        out = exp[exp["abs_amount"] >= thr].sort_values("abs_amount", ascending=False)

    return [
        {
            "date": pd.to_datetime(r["date"]).date().isoformat(),
            "payee": r.get("payee") or "(unknown)",
            "category": r.get("category") or "(uncategorized)",
            "amount": round(float(abs(r["amount"])), 2),
        }
        for _, r in out.iterrows()
    ]
