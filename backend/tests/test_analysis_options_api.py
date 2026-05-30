import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app
from backend.services.analysis_options import list_analysis_categories, list_analysis_payees


def test_list_analysis_categories_returns_sorted_expense_categories(monkeypatch):
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-05-01"), "amount": -40.0, "category_name": "Grocery", "payee": "King Soopers"},
            {"date": pd.Timestamp("2026-05-02"), "amount": -12.0, "category_name": "Dining", "payee": "Chipotle"},
            {"date": pd.Timestamp("2026-05-03"), "amount": 1200.0, "category_name": "Paycheck", "payee": "Employer"},
            {"date": pd.Timestamp("2026-05-04"), "amount": -8.0, "category_name": "Internal Transfer Expense", "payee": "Transfer"},
            {"date": pd.Timestamp("2026-05-05"), "amount": -18.0, "category_name": "Dining", "payee": "Sweetgreen"},
        ]
    )
    monkeypatch.setattr("backend.services.analysis_options.get_transactions_in_date_range", lambda *args, **kwargs: frame)

    categories = list_analysis_categories()

    assert categories == ["Dining", "Grocery"]



def test_list_analysis_payees_returns_ranked_recent_expense_payees(monkeypatch):
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-05-01"), "amount": -40.0, "category_name": "Grocery", "payee": "King Soopers"},
            {"date": pd.Timestamp("2026-05-02"), "amount": -12.0, "category_name": "Dining", "payee": "Chipotle"},
            {"date": pd.Timestamp("2026-05-03"), "amount": -55.0, "category_name": "Grocery", "payee": "King Soopers"},
            {"date": pd.Timestamp("2026-05-04"), "amount": -8.0, "category_name": "Internal Transfer Expense", "payee": "Transfer"},
            {"date": pd.Timestamp("2026-05-05"), "amount": 1200.0, "category_name": "Paycheck", "payee": "Employer"},
            {"date": pd.Timestamp("2026-05-06"), "amount": -33.0, "category_name": "Shopping", "payee": "Target"},
        ]
    )
    monkeypatch.setattr("backend.services.analysis_options.get_transactions_in_date_range", lambda *args, **kwargs: frame)

    payees = list_analysis_payees(limit=3)

    assert payees == ["King Soopers", "Target", "Chipotle"]



def test_analysis_category_options_endpoint_returns_items(monkeypatch):
    monkeypatch.setattr("backend.app.list_analysis_categories", lambda **kwargs: ["Bills", "Dining", "Grocery"])

    client = TestClient(app)
    response = client.get("/api/analysis/options/categories", params={"account_pid": "acct-1"})

    assert response.status_code == 200
    assert response.json() == {"items": ["Bills", "Dining", "Grocery"]}



def test_analysis_payee_options_endpoint_returns_items(monkeypatch):
    monkeypatch.setattr("backend.app.list_analysis_payees", lambda **kwargs: ["Amazon", "Costco Wholesale"])

    client = TestClient(app)
    response = client.get("/api/analysis/options/payees", params={"limit": 10})

    assert response.status_code == 200
    assert response.json() == {"items": ["Amazon", "Costco Wholesale"]}
