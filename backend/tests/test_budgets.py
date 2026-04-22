import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.services.budgets as budgets_service
from backend.services.budgets import (
    create_budget_plan,
    get_active_budget_plan,
    get_budget_status,
    get_category_budget_status,
    update_budget_target,
)


def test_create_budget_plan_persists_targets_and_sets_active_status(tmp_path):
    db_path = tmp_path / "planner.sqlite"

    plan = create_budget_plan(
        "2026-04-01",
        "2026-04-30",
        [
            {"category_name": "Grocery", "target_amount": 500},
            {"category_name": "Dining", "target_amount": 250},
        ],
        db_path=str(db_path),
    )

    assert plan["status"] == "active"
    assert [item["category_name"] for item in plan["targets"]] == ["Dining", "Grocery"]
    assert plan["targets"][0]["target_amount"] == 250.0
    assert plan["targets"][1]["target_amount"] == 500.0


def test_create_budget_plan_archives_previous_active_plan(tmp_path):
    db_path = tmp_path / "planner.sqlite"

    first_plan = create_budget_plan(
        "2026-04-01",
        "2026-04-30",
        [{"category_name": "Grocery", "target_amount": 500}],
        db_path=str(db_path),
    )
    second_plan = create_budget_plan(
        "2026-05-01",
        "2026-05-31",
        [{"category_name": "Grocery", "target_amount": 550}],
        db_path=str(db_path),
    )

    active_plan = get_active_budget_plan(db_path=str(db_path))

    assert active_plan is not None
    assert active_plan["plan_id"] == second_plan["plan_id"]

    archived_first = budgets_service.get_budget_plan(first_plan["plan_id"], db_path=str(db_path))
    assert archived_first["status"] == "archived"


def test_update_budget_target_upserts_target_amount(tmp_path):
    db_path = tmp_path / "planner.sqlite"
    plan = create_budget_plan(
        "2026-04-01",
        "2026-04-30",
        [{"category_name": "Grocery", "target_amount": 500}],
        db_path=str(db_path),
    )

    updated = update_budget_target(
        plan["plan_id"],
        "Grocery",
        650,
        db_path=str(db_path),
    )
    updated = update_budget_target(
        plan["plan_id"],
        "Dining",
        300,
        db_path=str(db_path),
    )

    target_map = {item["category_name"]: item["target_amount"] for item in updated["targets"]}
    assert target_map == {"Dining": 300.0, "Grocery": 650.0}


def test_get_budget_status_compares_targets_to_actual_spend(monkeypatch, tmp_path):
    db_path = tmp_path / "planner.sqlite"
    plan = create_budget_plan(
        "2026-04-01",
        "2026-04-30",
        [
            {"category_name": "Grocery", "target_amount": 500},
            {"category_name": "Dining", "target_amount": 250},
        ],
        db_path=str(db_path),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-03", "2026-04-04", "2026-04-05"]),
            "amount": [-120.0, -80.0, -40.0],
            "payee": ["Trader Joe's", "Whole Foods", "Cafe"],
            "category": ["Grocery", "Grocery", "Dining"],
            "account": ["Checking", "Checking", "Checking"],
        }
    )
    monkeypatch.setattr(
        budgets_service,
        "get_transactions_in_date_range",
        lambda *args, **kwargs: frame.copy(),
    )

    status = get_budget_status(plan["plan_id"], db_path=str(db_path))

    assert status["summary"] == {
        "total_target": 750.0,
        "total_actual": 240.0,
        "total_remaining": 510.0,
        "utilization_pct": 32.0,
    }
    category_map = {item["category_name"]: item for item in status["categories"]}
    assert category_map["Grocery"]["actual_amount"] == 200.0
    assert category_map["Grocery"]["remaining_amount"] == 300.0
    assert category_map["Dining"]["actual_amount"] == 40.0


def test_get_category_budget_status_marks_overspent(monkeypatch, tmp_path):
    db_path = tmp_path / "planner.sqlite"
    plan = create_budget_plan(
        "2026-04-01",
        "2026-04-30",
        [{"category_name": "Dining", "target_amount": 100}],
        db_path=str(db_path),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-03", "2026-04-05"]),
            "amount": [-75.0, -60.0],
            "payee": ["Restaurant A", "Restaurant B"],
            "category": ["Dining", "Dining"],
            "account": ["Checking", "Checking"],
        }
    )
    monkeypatch.setattr(
        budgets_service,
        "get_transactions_in_date_range",
        lambda *args, **kwargs: frame.copy(),
    )

    category_status = get_category_budget_status("Dining", plan["plan_id"], db_path=str(db_path))

    assert category_status["actual_amount"] == 135.0
    assert category_status["remaining_amount"] == -35.0
    assert category_status["status"] == "overspent"
