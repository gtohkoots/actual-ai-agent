import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.conversations import (
    append_message,
    load_conversation,
    load_planner_state,
    save_planner_state,
    update_conversation_context,
    upsert_conversation,
)


def test_save_planner_state_persists_under_existing_conversation_context(tmp_path):
    db_path = str(tmp_path / "chat.sqlite")

    upsert_conversation(
        "conv-planner",
        context={
            "account_pid": "acct-123",
            "account_name": "Visa",
            "card_label": "Visa",
            "selected_tab": "card",
        },
        db_path=db_path,
    )
    append_message("conv-planner", "user", "Create a budget", context={}, db_path=db_path)

    save_planner_state(
        "conv-planner",
        {
            "assistant_mode": "planner",
            "awaiting_approval": True,
            "pending_recommendation": {"period_start": "2026-05-06"},
        },
        db_path=db_path,
    )

    thread = load_conversation("conv-planner", db_path=db_path)

    assert thread["account_pid"] == "acct-123"
    assert thread["context"]["selected_tab"] == "card"
    assert thread["context"]["planner_state"]["assistant_mode"] == "planner"
    assert thread["context"]["planner_state"]["awaiting_approval"] is True


def test_update_conversation_context_merges_nested_planner_state(tmp_path):
    db_path = str(tmp_path / "chat.sqlite")

    upsert_conversation(
        "conv-merge",
        context={
            "account_pid": "acct-123",
            "planner_state": {
                "assistant_mode": "planner",
                "awaiting_approval": True,
            },
        },
        db_path=db_path,
    )

    update_conversation_context(
        "conv-merge",
        {
            "planner_state": {
                "latest_saved_plan": {"plan_id": "plan-1"},
            }
        },
        db_path=db_path,
    )

    planner_state = load_planner_state("conv-merge", db_path=db_path)

    assert planner_state["assistant_mode"] == "planner"
    assert planner_state["awaiting_approval"] is True
    assert planner_state["latest_saved_plan"]["plan_id"] == "plan-1"


def test_append_message_with_partial_context_does_not_drop_existing_planner_state(tmp_path):
    db_path = str(tmp_path / "chat.sqlite")

    upsert_conversation(
        "conv-history",
        context={
            "account_pid": "acct-123",
            "planner_state": {
                "awaiting_approval": True,
            },
        },
        db_path=db_path,
    )

    append_message(
        "conv-history",
        "assistant",
        "Please confirm the draft.",
        context={"selected_tab": "budget"},
        db_path=db_path,
    )

    thread = load_conversation("conv-history", db_path=db_path)
    assert thread["context"]["selected_tab"] == "budget"
    assert thread["context"]["planner_state"]["awaiting_approval"] is True
