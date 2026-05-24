import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis import tools as analysis_tools


EXPECTED_TOOL_NAMES = {
    "compare_periods",
    "detect_spending_anomalies",
    "find_recurring_charges",
    "find_similar_spending_weeks",
    "get_account_breakdown",
    "get_category_spend",
    "get_portfolio_summary",
    "get_recent_anomalies",
    "get_spending_drift",
    "get_transaction_slice",
    "search_documents",
    "search_past_weeks_by_category",
    "search_reports",
}


def test_list_analysis_tool_specs_exposes_expected_names():
    specs = analysis_tools.list_analysis_tool_specs()

    assert {item["name"] for item in specs} == EXPECTED_TOOL_NAMES
    assert all("executor" not in item for item in specs)


def test_execute_analysis_tool_routes_to_ledger_analysis(monkeypatch):
    seen = {}

    def fake_summary(period_start, period_end, **kwargs):
        seen["args"] = (period_start, period_end)
        seen["kwargs"] = kwargs
        return {"ok": True, "source": "ledger"}

    monkeypatch.setattr(analysis_tools.ledger_analysis, "get_portfolio_summary", fake_summary)
    monkeypatch.setitem(
        analysis_tools.TOOL_REGISTRY,
        "get_portfolio_summary",
        analysis_tools.AnalysisToolSpec(
            name="get_portfolio_summary",
            description="summary",
            arg_schema={},
            executor=analysis_tools.ledger_analysis.get_portfolio_summary,
        ),
    )

    result = analysis_tools.execute_analysis_tool(
        "get_portfolio_summary",
        period_start="2026-05-01",
        period_end="2026-05-24",
        account_name="Checking",
    )

    assert result == {"ok": True, "source": "ledger"}
    assert seen["args"] == ("2026-05-01", "2026-05-24")
    assert seen["kwargs"] == {"account_name": "Checking"}


def test_execute_analysis_tool_routes_to_document_services(monkeypatch):
    def fake_reports(query, **kwargs):
        return [{"query": query, "limit": kwargs.get("limit", 5)}]

    monkeypatch.setattr(analysis_tools.document_services, "search_reports", fake_reports)
    monkeypatch.setitem(
        analysis_tools.TOOL_REGISTRY,
        "search_reports",
        analysis_tools.AnalysisToolSpec(
            name="search_reports",
            description="reports",
            arg_schema={},
            executor=analysis_tools.document_services.search_reports,
        ),
    )

    result = analysis_tools.execute_analysis_tool("search_reports", query="Dining", limit=2)

    assert result == [{"query": "Dining", "limit": 2}]


def test_get_analysis_tool_spec_rejects_unknown_name():
    try:
        analysis_tools.get_analysis_tool_spec("missing_tool")
    except KeyError as exc:
        assert "Unknown analysis tool" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown analysis tool")
