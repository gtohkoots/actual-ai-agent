import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis import executor as analysis_executor
from backend.agents.analysis import tools as analysis_tools


@pytest.fixture
def registry_snapshot():
    snapshot = dict(analysis_tools.TOOL_REGISTRY)
    try:
        yield
    finally:
        analysis_tools.TOOL_REGISTRY.clear()
        analysis_tools.TOOL_REGISTRY.update(snapshot)


def test_execute_analysis_tool_plan_runs_tools_in_order(monkeypatch, registry_snapshot):
    calls = []

    def fake_summary(period_start, period_end):
        calls.append(("summary", period_start, period_end))
        return {"summary": True}

    def fake_category(period_start, period_end, limit=10):
        calls.append(("category", period_start, period_end, limit))
        return {"categories": []}

    monkeypatch.setitem(
        analysis_tools.TOOL_REGISTRY,
        "get_portfolio_summary",
        analysis_tools.AnalysisToolSpec("get_portfolio_summary", "summary", {"required": ["period_start", "period_end"], "properties": {"period_start": {}, "period_end": {}}}, fake_summary),
    )
    monkeypatch.setitem(
        analysis_tools.TOOL_REGISTRY,
        "get_category_spend",
        analysis_tools.AnalysisToolSpec("get_category_spend", "category", {"required": ["period_start", "period_end"], "properties": {"period_start": {}, "period_end": {}, "limit": {}}}, fake_category),
    )

    result = analysis_executor.execute_analysis_tool_plan(
        {
            "scope": "portfolio",
            "intent": "summary",
            "tool_calls": [
                {"tool": "get_portfolio_summary", "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"}},
                {"tool": "get_category_spend", "args": {"period_start": "2026-05-01", "period_end": "2026-05-24", "limit": 8}},
            ],
        }
    )

    assert calls == [
        ("summary", "2026-05-01", "2026-05-24"),
        ("category", "2026-05-01", "2026-05-24", 8),
    ]
    assert result["used_tools"] == ["get_portfolio_summary", "get_category_spend"]
    assert result["failures"] == []
    assert result["tool_results"][0]["step"] == 0
    assert result["tool_results"][1]["step"] == 1


def test_execute_analysis_tool_plan_collects_failures_and_continues(monkeypatch, registry_snapshot):
    def bad_tool(period_start, period_end):
        raise RuntimeError("boom")

    def good_tool(period_start, period_end):
        return {"ok": True}

    monkeypatch.setitem(
        analysis_tools.TOOL_REGISTRY,
        "detect_spending_anomalies",
        analysis_tools.AnalysisToolSpec("detect_spending_anomalies", "bad", {"required": ["period_start", "period_end"], "properties": {"period_start": {}, "period_end": {}}}, bad_tool),
    )
    monkeypatch.setitem(
        analysis_tools.TOOL_REGISTRY,
        "find_recurring_charges",
        analysis_tools.AnalysisToolSpec("find_recurring_charges", "good", {"required": ["period_start", "period_end"], "properties": {"period_start": {}, "period_end": {}}}, good_tool),
    )

    result = analysis_executor.execute_analysis_tool_plan(
        {
            "scope": "portfolio",
            "intent": "anomaly_review",
            "tool_calls": [
                {"tool": "detect_spending_anomalies", "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"}},
                {"tool": "find_recurring_charges", "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"}},
            ],
        }
    )

    assert result["used_tools"] == ["find_recurring_charges"]
    assert result["failures"] == [
        {
            "tool": "detect_spending_anomalies",
            "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"},
            "step": 0,
            "error": "boom",
        }
    ]
    assert result["tool_results"][0]["tool"] == "find_recurring_charges"


def test_execute_analysis_tool_plan_rejects_invalid_plan_shape():
    with pytest.raises(analysis_executor.AnalysisToolExecutionError):
        analysis_executor.execute_analysis_tool_plan({"tool_calls": []})


def test_execute_analysis_tool_plan_rejects_missing_required_args(registry_snapshot):
    with pytest.raises(analysis_executor.AnalysisToolExecutionError) as exc:
        analysis_executor.execute_analysis_tool_plan(
            {
                "tool_calls": [
                    {"tool": "get_portfolio_summary", "args": {"period_start": "2026-05-01"}},
                ]
            }
        )

    assert "missing required args" in str(exc.value)
