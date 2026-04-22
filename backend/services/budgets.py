from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional

from backend.services.filters import filter_internal_transfer_rows
from backend.utils.db import get_transactions_in_date_range
from backend.utils.planner_db import get_planner_connection


def _now_iso() -> str:
    """Return a timestamp string for planner-row audit fields."""
    return datetime.utcnow().isoformat()


def _normalize_targets(targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize incoming budget target payloads."""
    normalized: list[dict[str, Any]] = []
    for item in targets:
        category_name = str(item["category_name"]).strip()
        if not category_name:
            raise ValueError("Budget target category_name cannot be empty.")
        target_amount = round(float(item["target_amount"]), 2)
        if target_amount < 0:
            raise ValueError("Budget target amount cannot be negative.")
        normalized.append(
            {
                "category_name": category_name,
                "target_amount": target_amount,
            }
        )
    return normalized


def create_budget_plan(
    period_start: str,
    period_end: str,
    targets: Iterable[dict[str, Any]],
    *,
    status: str = "active",
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Create a budget plan with category targets and optionally make it active."""
    normalized_targets = _normalize_targets(targets)
    plan_id = str(uuid.uuid4())
    now = _now_iso()

    with get_planner_connection(db_path) as conn:
        # For the minimal planner setup we keep one active plan at a time and
        # archive any previous active plan when a new active one is created.
        if status == "active":
            conn.execute(
                """
                UPDATE budget_plans
                SET status = 'archived', updated_at = ?
                WHERE status = 'active'
                """,
                (now,),
            )

        conn.execute(
            """
            INSERT INTO budget_plans (plan_id, period_start, period_end, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (plan_id, period_start, period_end, status, now, now),
        )
        conn.executemany(
            """
            INSERT INTO budget_targets (plan_id, category_name, target_amount, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (plan_id, item["category_name"], item["target_amount"], now, now)
                for item in normalized_targets
            ],
        )

    return get_budget_plan(plan_id, db_path=db_path)


def get_budget_plan(plan_id: str, *, db_path: Optional[str] = None) -> dict[str, Any]:
    """Load one budget plan and all of its category targets."""
    with get_planner_connection(db_path) as conn:
        plan = conn.execute(
            """
            SELECT plan_id, period_start, period_end, status, created_at, updated_at
            FROM budget_plans
            WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        if plan is None:
            raise KeyError(plan_id)
        targets = conn.execute(
            """
            SELECT category_name, target_amount, created_at, updated_at
            FROM budget_targets
            WHERE plan_id = ?
            ORDER BY category_name ASC
            """,
            (plan_id,),
        ).fetchall()

    return {
        "plan_id": plan["plan_id"],
        "period_start": plan["period_start"],
        "period_end": plan["period_end"],
        "status": plan["status"],
        "created_at": plan["created_at"],
        "updated_at": plan["updated_at"],
        "targets": [
            {
                "category_name": row["category_name"],
                "target_amount": round(float(row["target_amount"]), 2),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in targets
        ],
    }


def get_active_budget_plan(*, db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return the most recently updated active budget plan, if one exists."""
    with get_planner_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT plan_id
            FROM budget_plans
            WHERE status = 'active'
            ORDER BY period_start DESC, updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    return get_budget_plan(row["plan_id"], db_path=db_path)


def update_budget_target(
    plan_id: str,
    category_name: str,
    target_amount: float,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Insert or update a category target within an existing budget plan."""
    normalized_category = str(category_name).strip()
    if not normalized_category:
        raise ValueError("category_name cannot be empty.")
    normalized_amount = round(float(target_amount), 2)
    if normalized_amount < 0:
        raise ValueError("target_amount cannot be negative.")

    now = _now_iso()
    with get_planner_connection(db_path) as conn:
        plan = conn.execute(
            "SELECT plan_id FROM budget_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if plan is None:
            raise KeyError(plan_id)
        conn.execute(
            """
            INSERT INTO budget_targets (plan_id, category_name, target_amount, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(plan_id, category_name)
            DO UPDATE SET target_amount = excluded.target_amount, updated_at = excluded.updated_at
            """,
            (plan_id, normalized_category, normalized_amount, now, now),
        )
        conn.execute(
            """
            UPDATE budget_plans
            SET updated_at = ?
            WHERE plan_id = ?
            """,
            (now, plan_id),
        )

    return get_budget_plan(plan_id, db_path=db_path)


def get_budget_status(
    plan_id: Optional[str] = None,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compare a budget plan's targets against live expense actuals for its period."""
    plan = get_budget_plan(plan_id, db_path=db_path) if plan_id else get_active_budget_plan(db_path=db_path)
    if plan is None:
        raise KeyError("No active budget plan was found.")

    # Budget status is computed from live ledger actuals, not cached values in
    # the planner DB, so the plan always compares against current transactions.
    actuals = _load_actuals_by_category(plan["period_start"], plan["period_end"], db_path=db_path)
    total_target = round(sum(item["target_amount"] for item in plan["targets"]), 2)
    total_actual = round(sum(actuals.values()), 2)
    total_remaining = round(total_target - total_actual, 2)

    categories = [
        _build_category_status(
            category_name=item["category_name"],
            target_amount=float(item["target_amount"]),
            actual_amount=float(actuals.get(item["category_name"], 0.0)),
            period_start=plan["period_start"],
            period_end=plan["period_end"],
        )
        for item in plan["targets"]
    ]
    categories.sort(key=lambda item: item["category_name"].lower())

    return {
        "plan_id": plan["plan_id"],
        "period_start": plan["period_start"],
        "period_end": plan["period_end"],
        "status": plan["status"],
        "summary": {
            "total_target": total_target,
            "total_actual": total_actual,
            "total_remaining": total_remaining,
            "utilization_pct": round((total_actual / total_target) * 100.0, 1) if total_target else 0.0,
        },
        "categories": categories,
    }


def get_category_budget_status(
    category_name: str,
    plan_id: Optional[str] = None,
    *,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return target-versus-actual status for one budgeted category."""
    plan = get_budget_plan(plan_id, db_path=db_path) if plan_id else get_active_budget_plan(db_path=db_path)
    if plan is None:
        raise KeyError("No active budget plan was found.")

    normalized_category = str(category_name).strip()
    if not normalized_category:
        raise ValueError("category_name cannot be empty.")

    target = next((item for item in plan["targets"] if item["category_name"] == normalized_category), None)
    if target is None:
        raise KeyError(normalized_category)

    actuals = _load_actuals_by_category(plan["period_start"], plan["period_end"], db_path=db_path)
    return _build_category_status(
        category_name=normalized_category,
        target_amount=float(target["target_amount"]),
        actual_amount=float(actuals.get(normalized_category, 0.0)),
        period_start=plan["period_start"],
        period_end=plan["period_end"],
    )


def _load_actuals_by_category(start_date: str, end_date: str, *, db_path: Optional[str] = None) -> dict[str, float]:
    """Aggregate live expense actuals by category for a planner time window."""
    frame = get_transactions_in_date_range(
        start_date,
        end_date,
        db_path=db_path,
        join_names=True,
        dollars=True,
        debug=False,
    )
    # Planner actuals should reflect true external spending behavior rather than
    # internal money movement between accounts.
    frame = filter_internal_transfer_rows(frame)
    if frame.empty:
        return {}

    expense_df = frame[frame["amount"] < 0].copy()
    if expense_df.empty:
        return {}

    grouped = (
        expense_df.groupby("category", dropna=False)["amount"]
        .sum()
        .mul(-1.0)
        .sort_values(ascending=False)
    )
    return {
        str(category) if category is not None else "(uncategorized)": round(float(amount), 2)
        for category, amount in grouped.items()
    }


def _build_category_status(
    *,
    category_name: str,
    target_amount: float,
    actual_amount: float,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    """Build one category's budget status record from target, actual, and time elapsed."""
    remaining = round(target_amount - actual_amount, 2)
    utilization_pct = round((actual_amount / target_amount) * 100.0, 1) if target_amount else 0.0

    elapsed_ratio = _elapsed_period_ratio(period_start, period_end)
    pace_ratio = (actual_amount / target_amount) if target_amount else 0.0

    # Use a simple first-pass status model: overspent if already above target,
    # at_risk if spend pace is materially ahead of elapsed time, else on_track.
    if actual_amount > target_amount:
        status = "overspent"
    elif target_amount and pace_ratio > elapsed_ratio * 1.1:
        status = "at_risk"
    else:
        status = "on_track"

    return {
        "category_name": category_name,
        "target_amount": round(target_amount, 2),
        "actual_amount": round(actual_amount, 2),
        "remaining_amount": remaining,
        "utilization_pct": utilization_pct,
        "elapsed_period_pct": round(elapsed_ratio * 100.0, 1),
        "status": status,
    }


def _elapsed_period_ratio(period_start: str, period_end: str) -> float:
    """Estimate how much of the budget period has elapsed as a 0..1 ratio."""
    start = datetime.strptime(period_start, "%Y-%m-%d").date()
    end = datetime.strptime(period_end, "%Y-%m-%d").date()
    today = date.today()
    if today <= start:
        return 0.0
    if today >= end:
        return 1.0

    total_days = (end - start).days + 1
    elapsed_days = (today - start).days + 1
    if total_days <= 0:
        return 1.0
    return max(0.0, min(1.0, elapsed_days / total_days))
