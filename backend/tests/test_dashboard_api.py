import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import app


def test_accounts_endpoint_exposes_account_metadata():
    client = TestClient(app)

    response = client.get("/api/accounts")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 1
    assert {"account_pid", "account_name", "transaction_count", "balance_current"} <= set(payload[0].keys())


def test_dashboard_endpoint_returns_account_summaries():
    client = TestClient(app)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert "month_label" in payload
    assert "selected_window" in payload
    assert "portfolio" in payload
    assert "accounts" in payload
    assert len(payload["accounts"]) >= 1

    account = payload["accounts"][0]
    assert {"account_pid", "account_name", "balance_current", "cycle_spend", "summary", "categories", "merchants", "transactions"} <= set(account.keys())
    assert "totalIncome" in account["summary"]
    assert account["context"]["accountPid"] == account["account_pid"]
    assert {"summary", "series", "categoryMix", "accountComparison", "topMovers", "dailyHeatmap", "topMerchants"} <= set(payload["portfolio"].keys())
    assert "totalBalance" in payload["portfolio"]["summary"]


def test_dashboard_default_window_uses_all_time_range():
    client = TestClient(app)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_window"] == "All time"
    assert payload["window"]["start"] <= payload["window"]["end"]
