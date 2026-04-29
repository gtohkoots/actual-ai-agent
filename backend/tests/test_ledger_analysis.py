import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.services.ledger_analysis as ledger_analysis


def test_get_portfolio_summary_filters_internal_transfers(monkeypatch):
    frame = pd.DataFrame(
        [
            {"date": "2026-04-01", "amount": 3000.0, "payee": "Payroll", "category_name": "Salary", "account_name": "Checking"},
            {"date": "2026-04-02", "amount": -200.0, "payee": "Grocer", "category_name": "Grocery", "account_name": "Checking"},
            {"date": "2026-04-03", "amount": -100.0, "payee": "Transfer", "category_name": "Internal Transfer Expense", "account_name": "Checking"},
        ]
    )
    monkeypatch.setattr(ledger_analysis, "get_transactions_in_date_range", lambda *args, **kwargs: frame.copy())

    result = ledger_analysis.get_portfolio_summary("2026-04-01", "2026-04-30")

    assert result["summary"] == {
        "total_income": 3000.0,
        "total_expense": 200.0,
        "net_cashflow": 2800.0,
        "transaction_count": 2,
    }


def test_get_category_spend_returns_ranked_categories(monkeypatch):
    frame = pd.DataFrame(
        [
            {"date": "2026-04-01", "amount": -200.0, "payee": "Grocer", "category_name": "Grocery", "account_name": "Checking"},
            {"date": "2026-04-02", "amount": -120.0, "payee": "Cafe", "category_name": "Dining", "account_name": "Checking"},
            {"date": "2026-04-03", "amount": -80.0, "payee": "Grocer", "category_name": "Grocery", "account_name": "Checking"},
        ]
    )
    monkeypatch.setattr(ledger_analysis, "get_transactions_in_date_range", lambda *args, **kwargs: frame.copy())

    result = ledger_analysis.get_category_spend("2026-04-01", "2026-04-30")

    assert result["total_expense"] == 400.0
    assert result["categories"][0]["category_name"] == "Grocery"
    assert result["categories"][0]["amount"] == 280.0
    assert result["categories"][1]["category_name"] == "Dining"


def test_compare_periods_returns_total_and_category_deltas(monkeypatch):
    current = pd.DataFrame(
        [
            {"date": "2026-04-01", "amount": -200.0, "payee": "Grocer", "category_name": "Grocery", "account_name": "Checking"},
            {"date": "2026-04-02", "amount": -150.0, "payee": "Cafe", "category_name": "Dining", "account_name": "Checking"},
        ]
    )
    previous = pd.DataFrame(
        [
            {"date": "2026-03-01", "amount": -100.0, "payee": "Grocer", "category_name": "Grocery", "account_name": "Checking"},
            {"date": "2026-03-02", "amount": -50.0, "payee": "Cafe", "category_name": "Dining", "account_name": "Checking"},
        ]
    )

    def fake_loader(start_date, end_date, **kwargs):
        if start_date == "2026-04-01":
            return current.copy()
        return previous.copy()

    monkeypatch.setattr(ledger_analysis, "get_transactions_in_date_range", fake_loader)

    result = ledger_analysis.compare_periods("2026-04-01", "2026-04-30", "2026-03-01", "2026-03-31")

    assert result["total_deltas"]["total_expense"] == 200.0
    assert result["category_changes"][0]["delta"] == 100.0


def test_get_transaction_slice_applies_filters(monkeypatch):
    frame = pd.DataFrame(
        [
            {"date": "2026-04-01", "amount": -200.0, "payee": "Grocer", "category_name": "Grocery", "account_name": "Checking"},
            {"date": "2026-04-02", "amount": -50.0, "payee": "Cafe", "category_name": "Dining", "account_name": "Checking"},
        ]
    )
    monkeypatch.setattr(ledger_analysis, "get_transactions_in_date_range", lambda *args, **kwargs: frame.copy())

    result = ledger_analysis.get_transaction_slice(
        "2026-04-01",
        "2026-04-30",
        category_name="Dining",
    )

    assert len(result["transactions"]) == 1
    assert result["transactions"][0]["category_name"] == "Dining"
