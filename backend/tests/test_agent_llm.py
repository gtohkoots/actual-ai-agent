import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.agents.llm as planner_llm


def test_generate_planner_response_uses_deterministic_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = planner_llm.generate_planner_response(
        {
            "user_message": "Review my budget",
            "active_budget_plan": {"status": "missing"},
            "budget_status": {"status": "missing"},
        }
    )

    assert result["summary"] == "No active budget plan is set up yet."
    assert result["next_action"] == "Create a budget plan before requesting a budget review."


def test_generate_planner_response_parses_model_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        content = """
        {
          "summary": "Budget is stable overall.",
          "highlights": ["Dining is slightly elevated."],
          "next_action": "Watch dining spend for the rest of the week."
        }
        """

    class FakeChatOpenAI:
        def __init__(self, model: str, temperature: int):
            self.model = model
            self.temperature = temperature

        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr(planner_llm, "ChatOpenAI", FakeChatOpenAI)

    result = planner_llm.generate_planner_response(
        {
            "user_message": "Review my budget",
            "active_budget_plan": {"status": "active"},
            "budget_status": {"status": "active", "summary": {}, "categories": []},
        }
    )

    assert result == {
        "summary": "Budget is stable overall.",
        "highlights": ["Dining is slightly elevated."],
        "next_action": "Watch dining spend for the rest of the week.",
    }


def test_interpret_planner_turn_intent_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = planner_llm.interpret_planner_turn_intent(
        "Create a budget starting today for a month and save $500",
        has_pending_recommendation=False,
    )

    assert result["intent"] == "budget_recommendation"
    assert result["allowed_tools"] == ["recommend_budget_targets"]


def test_interpret_planner_turn_intent_detects_revision_when_pending_draft_exists(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = planner_llm.interpret_planner_turn_intent(
        "Keep savings at $500 and increase Dine a bit.",
        has_pending_recommendation=True,
    )

    assert result["intent"] == "budget_revision"
    assert result["needs_pending_recommendation"] is True
    assert result["allowed_tools"] == ["revise_budget_recommendation"]


def test_interpret_planner_turn_intent_normalizes_model_tool_aliases(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        content = """
        {
          "intent": "budget_revision",
          "confidence": 0.91,
          "needs_pending_recommendation": true,
          "allowed_tools": ["revise_budget_targets"],
          "notes": "Revision request."
        }
        """

    class FakeChatOpenAI:
        def __init__(self, model: str, temperature: int):
            self.model = model
            self.temperature = temperature

        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr(planner_llm, "ChatOpenAI", FakeChatOpenAI)

    result = planner_llm.interpret_planner_turn_intent(
        "Increase Dine a bit.",
        has_pending_recommendation=True,
    )

    assert result["intent"] == "budget_revision"
    assert result["allowed_tools"] == ["revise_budget_recommendation"]


def test_interpret_planner_turn_intent_defaults_tools_when_model_returns_unsupported_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        content = """
        {
          "intent": "budget_revision",
          "confidence": 0.91,
          "needs_pending_recommendation": true,
          "allowed_tools": ["unknown_tool_name"],
          "notes": "Revision request."
        }
        """

    class FakeChatOpenAI:
        def __init__(self, model: str, temperature: int):
            self.model = model
            self.temperature = temperature

        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr(planner_llm, "ChatOpenAI", FakeChatOpenAI)

    result = planner_llm.interpret_planner_turn_intent(
        "Increase Dine a bit.",
        has_pending_recommendation=True,
    )

    assert result["intent"] == "budget_revision"
    assert result["allowed_tools"] == ["revise_budget_recommendation"]


def test_interpret_budget_request_parameters_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        planner_llm,
        "date",
        type(
            "FakeDate",
            (),
            {
                "today": staticmethod(lambda: __import__("datetime").date(2026, 5, 10)),
                "fromisoformat": staticmethod(__import__("datetime").date.fromisoformat),
            },
        ),
    )

    result = planner_llm.interpret_budget_request_parameters(
        "Create a budget starting today for a month and save $500",
    )

    assert result["period_start"] == "2026-05-10"
    assert result["period_end"] == "2026-06-08"
    assert result["savings_target"] == 500.0


def test_interpret_budget_request_parameters_supports_explicit_iso_range_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = planner_llm.interpret_budget_request_parameters(
        "Create a budget from 2026-04-01 to 2026-04-30 and save $700",
    )

    assert result["period_start"] == "2026-04-01"
    assert result["period_end"] == "2026-04-30"
    assert result["savings_target"] == 700.0


def test_interpret_budget_request_parameters_parses_model_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        content = """
        {
          "period_start": "2026-04-01",
          "period_end": "2026-04-30",
          "savings_target": 650,
          "notes": "User asked for April."
        }
        """

    class FakeChatOpenAI:
        def __init__(self, model: str, temperature: int):
            self.model = model
            self.temperature = temperature

        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr(planner_llm, "ChatOpenAI", FakeChatOpenAI)

    result = planner_llm.interpret_budget_request_parameters(
        "Create a budget for April 2026 and save $650",
    )

    assert result["period_start"] == "2026-04-01"
    assert result["period_end"] == "2026-04-30"
    assert result["savings_target"] == 650.0
