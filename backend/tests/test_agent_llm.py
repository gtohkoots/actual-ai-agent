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
