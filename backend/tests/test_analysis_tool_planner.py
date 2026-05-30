import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis.tool_planner import _normalize_tool_plan, plan_analysis_tool_calls


def _request(message="How is my spending trending?", **context_overrides):
    defaults = {
        "selected_tab": "overview",
        "account_pid": None,
        "account_name": None,
        "card_label": None,
        "start_date": "2026-05-01",
        "end_date": "2026-05-24",
        "scope_type": None,
        "selected_category": None,
        "selected_payee": None,
    }
    defaults.update(context_overrides)
    context = SimpleNamespace(**defaults)
    context.model_dump = lambda: dict(defaults)
    return SimpleNamespace(message=message, context=context)


def test_plan_analysis_tool_calls_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    request = _request()
    analysis_request = {"scope": "portfolio", "intent": "trend_review", "entity": {}}

    result = plan_analysis_tool_calls(request, analysis_request)

    assert result["planning_mode"] == "fallback"
    assert result["tool_calls"][0]["tool"] == "get_spending_drift"
    assert result["tool_calls"][1]["tool"] == "get_category_spend"


def test_plan_analysis_tool_calls_builds_category_deep_dive_from_context(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    request = _request(message="Review grocery", selected_category="Grocery", account_name="Visa", account_pid="acct-1")
    analysis_request = {"scope": "category", "intent": "category_deep_dive", "entity": {"category": "Grocery"}}

    result = plan_analysis_tool_calls(request, analysis_request)

    assert result["tool_calls"][0]["tool"] == "get_category_spend"
    assert result["tool_calls"][0]["args"]["account_name"] == "Visa"
    assert result["tool_calls"][1]["tool"] == "get_transaction_slice"
    assert result["tool_calls"][1]["args"]["category_name"] == "Grocery"
    assert result["tool_calls"][1]["args"]["account_name"] == "Visa"
    assert result["tool_calls"][2]["tool"] == "search_past_weeks_by_category"


def test_plan_analysis_tool_calls_builds_payee_scope_from_explicit_selection(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    request = _request(message="How is Costco trending?", selected_payee="Costco Wholesale", account_name="Visa", account_pid="acct-1")
    analysis_request = {"scope": "payee", "intent": "trend_review", "entity": {"payee": "Costco Wholesale"}}

    result = plan_analysis_tool_calls(request, analysis_request)

    assert result["tool_calls"][0]["tool"] == "search_reports"
    assert result["tool_calls"][0]["args"]["query"] == "Costco Wholesale"
    assert result["tool_calls"][1]["tool"] == "get_transaction_slice"
    assert result["tool_calls"][1]["args"]["payee"] == "Costco Wholesale"
    assert result["tool_calls"][1]["args"]["account_name"] == "Visa"


def test_plan_analysis_tool_calls_builds_payee_anomaly_plan(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    request = _request(message="Any unusual Costco charges?", selected_payee="Costco Wholesale")
    analysis_request = {"scope": "payee", "intent": "anomaly_review", "entity": {"payee": "Costco Wholesale"}}

    result = plan_analysis_tool_calls(request, analysis_request)

    assert result["tool_calls"][0]["tool"] == "detect_spending_anomalies"
    assert result["tool_calls"][1] == {"tool": "get_recent_anomalies", "args": {"payee": "Costco Wholesale", "limit": 5}}
    assert result["tool_calls"][2]["tool"] == "get_transaction_slice"


def test_normalize_tool_plan_filters_unknown_tools_and_args():
    request = _request()
    analysis_request = {"scope": "portfolio", "intent": "summary", "entity": {}}
    payload = {
        "scope": "portfolio",
        "intent": "summary",
        "reasoning": "Need summary and category mix.",
        "tool_calls": [
            {"tool": "get_portfolio_summary", "args": {"period_start": "2026-05-01", "period_end": "2026-05-24", "bogus": 1}},
            {"tool": "missing_tool", "args": {}},
        ],
    }

    result = _normalize_tool_plan(request, analysis_request, payload)

    assert result["planning_mode"] == "model"
    assert result["tool_calls"] == [
        {
            "tool": "get_portfolio_summary",
            "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"},
        }
    ]



def test_normalize_tool_plan_falls_back_when_required_args_missing():
    request = _request()
    analysis_request = {"scope": "portfolio", "intent": "summary", "entity": {}}
    payload = {
        "tool_calls": [
            {"tool": "get_portfolio_summary", "args": {"period_start": "2026-05-01"}},
        ]
    }

    result = _normalize_tool_plan(request, analysis_request, payload)

    assert result["planning_mode"] == "fallback"
    assert result["tool_calls"][0]["tool"] == "get_portfolio_summary"
