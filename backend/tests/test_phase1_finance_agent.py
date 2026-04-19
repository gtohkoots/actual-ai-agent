import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.services.dashboard as dashboard_service
from backend.langchain_runner import get_weekly_data_tool
from backend.services.insights import get_week_rollups, save_daily_snapshot
from backend.utils.db import get_transactions_in_date_range


def test_get_transactions_in_date_range_converts_amounts_and_exposes_aliases():
    df = get_transactions_in_date_range("2026-03-16", "2026-03-16", debug=False)

    assert "category" in df.columns
    assert "account" in df.columns
    assert df["amount"].dtype.kind == "f"
    assert df.iloc[0]["amount"] == -178.73


def test_week_rollups_use_human_category_names():
    payload = get_week_rollups("2026-03-16", "2026-03-22")
    categories = {item["category"] for item in payload["by_category"]}

    assert "nan" not in categories
    assert "(uncategorized)" not in categories
    assert "Grocery" in categories
    assert "Dine" in categories


def test_week_rollups_exclude_internal_transfer_categories():
    df = pd.DataFrame(
        {
            "date": ["2026-03-16", "2026-03-16", "2026-03-16", "2026-03-16"],
            "amount": [100.0, -20.0, 50.0, -50.0],
            "payee": ["Employer", "Grocer", "Checking Transfer", "Credit Payment"],
            "category_name": [
                "Paycheck",
                "Groceries",
                "Internal Transfer Income",
                "Internal Transfer Expense",
            ],
        }
    )

    payload = get_week_rollups("2026-03-16", "2026-03-16", df=df)

    assert payload["summary"]["total_income"] == 100.0
    assert payload["summary"]["total_expense"] == 20.0
    assert payload["summary"]["net_cashflow"] == 80.0
    assert all(item["category"] not in {"Internal Transfer Income", "Internal Transfer Expense"} for item in payload["by_category"])
    assert all(item["payee"] not in {"Checking Transfer", "Credit Payment"} for item in payload["top_payees"])
    assert all(item["payee"] != "Checking Transfer" for item in payload["income_payee_distribution"])


def test_get_weekly_data_tool_returns_dollar_amounts_and_category_fields():
    payload = json.loads(
        get_weekly_data_tool.invoke({"start_date": "2026-03-16", "end_date": "2026-03-16"})
    )

    assert payload
    assert payload[0]["amount"] == -178.73
    assert "category" in payload[0]
    assert "account" in payload[0]


def test_filter_internal_transfer_rows_excludes_transfer_categories():
    df = pd.DataFrame(
        {
            "category_name": ["Internal Transfer Expense", "Internal Transfer Income", "Food"],
            "amount": [-10.0, 10.0, -5.0],
        }
    )

    from backend.services.filters import filter_internal_transfer_rows, is_internal_transfer_category

    filtered = filter_internal_transfer_rows(df)

    assert filtered["category_name"].tolist() == ["Food"]
    assert is_internal_transfer_category("Internal Transfer Expense")
    assert is_internal_transfer_category("Internal Transfer Income")
    assert is_internal_transfer_category(" internal   transfer income ")


def test_dashboard_account_summary_keeps_internal_transfers_in_transactions(monkeypatch):
    current_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-16", "2026-03-17", "2026-03-18"]),
            "amount": [-20.0, 50.0, -50.0],
            "payee": ["Grocer", "Checking Transfer", "Credit Payment"],
            "category": ["Groceries", "Internal Transfer Income", "Internal Transfer Expense"],
            "account": ["Visa", "Visa", "Visa"],
        }
    )
    previous_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-09"]),
            "amount": [-10.0],
            "payee": ["Cafe"],
            "category": ["Dining"],
            "account": ["Visa"],
        }
    )

    def fake_get_transactions_in_date_range(start_date, end_date, **kwargs):
        if start_date == "2026-03-16":
            return current_df.copy()
        return previous_df.copy()

    monkeypatch.setattr(dashboard_service, "get_transactions_in_date_range", fake_get_transactions_in_date_range)

    summary = dashboard_service._build_summary_for_account(
        {
            "account_pid": "acct-1",
            "account_name": "Visa",
            "balance_current": 1000.0,
        },
        "2026-03-16",
        "2026-03-22",
    )

    assert summary.cycle_income == 0.0
    assert summary.cycle_spend == 20.0
    assert {item["merchant"] for item in summary.transactions} == {"Grocer", "Checking Transfer", "Credit Payment"}


def test_portfolio_income_mix_groups_income_by_category(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19"]),
            "amount": [1000.0, 250.0, 150.0, -40.0],
            "payee": ["Employer A", "Employer B", "Interest Provider", "Grocer"],
            "category": ["Salary", "Salary", "Interest", "Groceries"],
            "account": ["Checking", "Checking", "Checking", "Checking"],
        }
    )

    monkeypatch.setattr(
        dashboard_service,
        "get_transactions_in_date_range",
        lambda *args, **kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        dashboard_service,
        "list_accounts",
        lambda db_path=None: [{"account_pid": "acct-1", "account_name": "Checking", "balance_current": 5000.0}],
    )

    portfolio = dashboard_service._portfolio_overview("2026-03-16", "2026-03-22")

    assert portfolio["incomeMix"] == [
        {"source": "Salary", "amount": 1250.0, "share": 89.3},
        {"source": "Interest", "amount": 150.0, "share": 10.7},
    ]


def test_save_daily_snapshot_returns_empty_message_for_dates_without_transactions():
    result = save_daily_snapshot("1999-01-01", 200.0)

    assert result == "1999-01-01 无交易数据，未生成快照。"
