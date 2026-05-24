import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.analysis.intent import interpret_analysis_request
from backend.services.chat import ChatContext, ChatRequest


def test_interpret_analysis_request_defaults_to_portfolio_summary():
    request = ChatRequest(message="How are my finances looking this month?")

    result = interpret_analysis_request(request)

    assert result["scope"] == "portfolio"
    assert result["intent"] in {"summary", "trend_review"}


def test_interpret_analysis_request_prefers_category_scope_when_focus_category_exists():
    request = ChatRequest(
        message="What changed here?",
        context=ChatContext(focus_category="Grocery", account_name="Visa", card_label="Visa"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "category"
    assert result["entity"]["category"] == "grocery"


def test_interpret_analysis_request_detects_account_scope_when_message_mentions_card():
    request = ChatRequest(
        message="Compare this card to last week",
        context=ChatContext(account_pid="acct-1", account_name="Visa", card_label="Visa"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "account"
    assert result["intent"] == "comparison"
