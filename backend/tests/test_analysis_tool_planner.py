import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis.tool_planner import _normalize_tool_plan, plan_analysis_tool_calls


def _request(message="How is my spending trending?", **context_overrides):
    context = SimpleNamespace(
        selected_tab="overview",
        account_pid=None,
        account_name=None,
        card_label=None,
        start_date="2026-05-01",
        end_date="2026-05-24",
        focus_category=context_overrides.pop("focus_category", None),
        focus_payee=context_overrides.pop("focus_payee", None),
        model_dump=lambda: {
            "selected_tab": "overview",
            "account_pid": None,
            "account_name": None,
            "card_label": None,
            "start_date": "2026-05-01",
            "end_date": "2026-05-24",
            "focus_category": context_overrides.get("focus_category"),
            "focus_payee": context_overrides.get("focus_payee"),
        },
    )
    for key, value in context_overrides.items():
        setattr(context, key, value)
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
    request = _request(message="Review grocery", focus_category="Grocery")
    analysis_request = {"scope": "category", "intent": "category_deep_dive", "entity": {"category": "Grocery"}}

    result = plan_analysis_tool_calls(request, analysis_request)

    assert result["tool_calls"][1]["tool"] == "get_transaction_slice"
    assert result["tool_calls"][1]["args"]["category_name"] == "Grocery"
    assert result["tool_calls"][2]["tool"] == "search_past_weeks_by_category"


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
