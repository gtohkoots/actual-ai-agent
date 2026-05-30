import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis.llm import fallback_analysis_response, generate_analysis_response


def _request():
    return SimpleNamespace(
        message="How is my spending trending?",
        context=SimpleNamespace(
            card_label="Visa",
            account_name="Visa",
            selected_category=None,
            selected_payee=None,
            model_dump=lambda: {"card_label": "Visa"},
        ),
    )


def _tool_plan():
    return {
        "scope": "account",
        "intent": "comparison",
        "tool_calls": [{"tool": "get_portfolio_summary", "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"}}],
        "planning_mode": "fallback",
        "reasoning": "test",
    }


def _execution_result():
    return {
        "tool_results": [
            {
                "tool": "get_portfolio_summary",
                "args": {"period_start": "2026-05-01", "period_end": "2026-05-24"},
                "step": 0,
                "result": {
                    "period_start": "2026-05-01",
                    "period_end": "2026-05-24",
                    "summary": {"total_income": 3000.0, "total_expense": 1200.0, "net_cashflow": 1800.0},
                },
            }
        ],
        "used_tools": ["get_portfolio_summary"],
        "failures": [],
    }


def test_fallback_analysis_response_returns_structured_sections():
    result = fallback_analysis_response(
        _request(),
        _tool_plan(),
        _execution_result(),
        analysis_request={"scope": "account", "intent": "comparison", "entity": {}},
        reason="Model unavailable.",
    )

    assert result["summary"]
    assert result["findings"]
    assert result["content"]
    assert "**Findings**" in result["content"]
    assert result["retrieval_strategy"] == ["get_portfolio_summary"]


def test_generate_analysis_response_renders_model_payload(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            return SimpleNamespace(
                content='''{
                    "summary": "Spending is stable overall.",
                    "findings": ["Income exceeded expenses by $1,800.00."],
                    "risks": ["No major risk signals were found."],
                    "opportunities": ["Review categories next for more detail."],
                    "actions": ["Review top categories"],
                    "sources": [{"label": "get_portfolio_summary", "detail": "2026-05-01 to 2026-05-24 · Income $3000.00"}]
                }'''
            )

    monkeypatch.setattr("backend.agents.analysis.llm.ChatOpenAI", FakeChatOpenAI)

    result = generate_analysis_response(
        _request(),
        _tool_plan(),
        _execution_result(),
        analysis_request={"scope": "account", "intent": "comparison", "entity": {}},
    )

    assert result["summary"] == "Spending is stable overall."
    assert result["findings"] == ["Income exceeded expenses by $1,800.00."]
    assert result["content"]
    assert "**Findings**" in result["content"]
    assert result["actions"] == ["Review top categories"]
