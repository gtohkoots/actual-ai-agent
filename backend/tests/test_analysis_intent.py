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


def test_interpret_analysis_request_detects_category_scope_from_message():
    request = ChatRequest(
        message="Explain grocery spending this month",
        context=ChatContext(account_name="Visa", card_label="Visa"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "category"
    assert result["intent"] in {"category_deep_dive", "trend_review"}
    assert result["entity"]["category"] is None


def test_interpret_analysis_request_detects_account_scope_when_message_mentions_card():
    request = ChatRequest(
        message="Compare this card to last week",
        context=ChatContext(account_pid="acct-1", account_name="Visa", card_label="Visa"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "account"
    assert result["intent"] == "comparison"


def test_interpret_analysis_request_detects_payee_scope_from_message_only():
    request = ChatRequest(
        message="Why is Costco so high this month?",
        context=ChatContext(account_name="Visa", card_label="Visa"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "payee"
    assert result["entity"]["payee"] is None


def test_interpret_analysis_request_respects_explicit_portfolio_scope():
    request = ChatRequest(
        message="Why is Costco trending?",
        context=ChatContext(
            scope_type="portfolio",
            selected_payee="Costco Wholesale",
            account_name="Visa",
            card_label="Visa",
        ),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "portfolio"
    assert result["entity"] == {}


def test_interpret_analysis_request_respects_explicit_account_scope():
    request = ChatRequest(
        message="How is my spending doing?",
        context=ChatContext(
            scope_type="account",
            selected_account_pid="acct-1",
            selected_account_name="Visa",
            card_label="Visa",
        ),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "account"
    assert result["entity"]["account_pid"] == "acct-1"


def test_interpret_analysis_request_respects_explicit_category_scope():
    request = ChatRequest(
        message="How is this trending?",
        context=ChatContext(
            scope_type="category",
            selected_category="Grocery",
        ),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "category"
    assert result["entity"]["category"] == "grocery"


def test_interpret_analysis_request_respects_explicit_payee_scope():
    request = ChatRequest(
        message="How is this trending?",
        context=ChatContext(
            scope_type="payee",
            selected_payee="Costco Wholesale",
        ),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "payee"
    assert result["entity"]["payee"] == "costco wholesale"


def test_interpret_analysis_request_downgrades_incomplete_explicit_category_scope():
    request = ChatRequest(
        message="How is this trending?",
        context=ChatContext(scope_type="category"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "portfolio"
    assert result["entity"] == {}


def test_interpret_analysis_request_downgrades_incomplete_explicit_payee_scope():
    request = ChatRequest(
        message="How is this trending?",
        context=ChatContext(scope_type="payee"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "portfolio"
    assert result["entity"] == {}


def test_interpret_analysis_request_downgrades_incomplete_explicit_account_scope():
    request = ChatRequest(
        message="How is this account doing?",
        context=ChatContext(scope_type="account"),
    )

    result = interpret_analysis_request(request)

    assert result["scope"] == "portfolio"
    assert result["entity"] == {}
