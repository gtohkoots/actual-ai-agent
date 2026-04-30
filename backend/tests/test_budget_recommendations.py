import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.services.budget_recommendations as recommendation_service


def test_recommend_budget_targets_uses_weighted_history_and_default_savings(monkeypatch):
    frames = {
        ("2026-03-29", "2026-04-27"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"]),
                "amount": [3000.0, -1200.0, -500.0, -300.0, -350.0],
                "payee": ["Payroll", "Utility", "Grocer", "Cafe", "Savings Transfer"],
                "category_name": ["Paycheck", "Bills", "Grocery", "Dine", "Savings"],
            }
        ),
        ("2026-02-27", "2026-03-28"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"]),
                "amount": [3000.0, -1200.0, -450.0, -250.0, -320.0],
                "payee": ["Payroll", "Utility", "Grocer", "Cafe", "Savings Transfer"],
                "category_name": ["Paycheck", "Bills", "Grocery", "Dine", "Savings"],
            }
        ),
        ("2026-01-28", "2026-02-26"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]),
                "amount": [3000.0, -1200.0, -400.0, -200.0, -300.0],
                "payee": ["Payroll", "Utility", "Grocer", "Cafe", "Savings Transfer"],
                "category_name": ["Paycheck", "Bills", "Grocery", "Dine", "Savings"],
            }
        ),
    }

    monkeypatch.setattr(
        recommendation_service,
        "get_transactions_in_date_range",
        lambda start, end, **kwargs: frames[(start, end)].copy(),
    )

    result = recommendation_service.recommend_budget_targets("2026-04-28", "2026-05-27")

    assert result["expected_income"] == 3000.0
    assert result["planned_savings"] == 331.0
    assert result["historical_savings_baseline"] == 331.0
    category_map = {item["category_name"]: item for item in result["category_targets"]}
    assert category_map["Bills"]["recommended_target"] == 1200.0
    assert category_map["Grocery"]["recommended_target"] > category_map["Grocery"]["baseline_amount"]
    assert category_map["Dine"]["recommended_target"] < category_map["Dine"]["baseline_amount"]


def test_recommend_budget_targets_rebalances_discretionary_spend_for_large_savings_target(monkeypatch):
    frames = {
        ("2026-03-29", "2026-04-27"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"]),
                "amount": [2500.0, -1000.0, -600.0, -500.0, -400.0],
                "payee": ["Payroll", "Utility", "Grocer", "Cafe", "Store"],
                "category_name": ["Paycheck", "Bills", "Grocery", "Dine", "Shopping"],
            }
        ),
        ("2026-02-27", "2026-03-28"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"]),
                "amount": [2500.0, -1000.0, -600.0, -450.0, -350.0],
                "payee": ["Payroll", "Utility", "Grocer", "Cafe", "Store"],
                "category_name": ["Paycheck", "Bills", "Grocery", "Dine", "Shopping"],
            }
        ),
        ("2026-01-28", "2026-02-26"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05"]),
                "amount": [2500.0, -1000.0, -550.0, -400.0, -300.0],
                "payee": ["Payroll", "Utility", "Grocer", "Cafe", "Store"],
                "category_name": ["Paycheck", "Bills", "Grocery", "Dine", "Shopping"],
            }
        ),
    }

    monkeypatch.setattr(
        recommendation_service,
        "get_transactions_in_date_range",
        lambda start, end, **kwargs: frames[(start, end)].copy(),
    )

    result = recommendation_service.recommend_budget_targets(
        "2026-04-28",
        "2026-05-27",
        savings_target=800.0,
    )

    category_map = {item["category_name"]: item for item in result["category_targets"]}
    assert result["planned_savings"] == 800.0
    assert result["total_budgeted_spend"] <= 1700.0
    assert category_map["Dine"]["recommended_target"] < category_map["Dine"]["baseline_amount"]
    assert "Reduced to protect the savings target" in category_map["Dine"]["adjustment_reason"]


def test_recommend_budget_targets_ignores_internal_transfer_categories(monkeypatch):
    frames = {
        ("2026-03-29", "2026-04-27"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04"]),
                "amount": [3000.0, 500.0, -200.0, -150.0],
                "payee": ["Payroll", "Transfer In", "Transfer Out", "Grocer"],
                "category_name": ["Paycheck", "Internal Transfer Income", "Internal Transfer Expense", "Grocery"],
            }
        ),
        ("2026-02-27", "2026-03-28"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"]),
                "amount": [3000.0, 500.0, -200.0, -100.0],
                "payee": ["Payroll", "Transfer In", "Transfer Out", "Grocer"],
                "category_name": ["Paycheck", "Internal Transfer Income", "Internal Transfer Expense", "Grocery"],
            }
        ),
        ("2026-01-28", "2026-02-26"): pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03", "2026-02-04"]),
                "amount": [3000.0, 500.0, -200.0, -80.0],
                "payee": ["Payroll", "Transfer In", "Transfer Out", "Grocer"],
                "category_name": ["Paycheck", "Internal Transfer Income", "Internal Transfer Expense", "Grocery"],
            }
        ),
    }

    monkeypatch.setattr(
        recommendation_service,
        "get_transactions_in_date_range",
        lambda start, end, **kwargs: frames[(start, end)].copy(),
    )

    result = recommendation_service.recommend_budget_targets("2026-04-28", "2026-05-27")

    assert result["expected_income"] == 3000.0
    category_map = {item["category_name"]: item for item in result["category_targets"]}
    assert "Internal Transfer Expense" not in category_map
    assert "Internal Transfer Income" not in result["assumptions"]["income_categories_used"]
