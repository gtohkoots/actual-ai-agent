import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.mcp.prompts as mcp_prompts


def test_register_prompts_registers_budget_prompt_names():
    registered_prompts = []

    class FakeFastMCP:
        def prompt(self, **kwargs):
            def decorator(fn):
                registered_prompts.append(kwargs.get("name"))
                return fn
            return decorator

    mcp_prompts.register_prompts(FakeFastMCP())

    assert registered_prompts == [
        "review_current_budget",
        "adjust_budget_target",
        "recommend_budget_plan",
        "revise_budget_plan",
        "finalize_budget_plan",
    ]


def test_review_current_budget_prompt_includes_budget_context(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def prompt(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    monkeypatch.setattr(mcp_prompts, "read_active_budget_plan", lambda db_path=None: {"plan_id": "plan-1"})
    monkeypatch.setattr(mcp_prompts, "read_current_budget_status", lambda db_path=None: {"summary": {"total_target": 1000.0}})

    mcp_prompts.register_prompts(FakeFastMCP())

    prompt_text = registered["review_current_budget"]()
    assert "plan-1" in prompt_text
    assert "total_target" in prompt_text


def test_adjust_budget_target_prompt_includes_requested_change(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def prompt(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    monkeypatch.setattr(mcp_prompts, "read_active_budget_plan", lambda db_path=None: {"plan_id": "plan-1"})
    monkeypatch.setattr(mcp_prompts, "read_current_budget_status", lambda db_path=None: {"summary": {"total_actual": 200.0}})

    mcp_prompts.register_prompts(FakeFastMCP())

    prompt_text = registered["adjust_budget_target"]("Dining", 300.0)
    assert "Dining" in prompt_text
    assert "300.0" in prompt_text
    assert "plan-1" in prompt_text


def test_recommend_budget_plan_prompt_includes_recommended_payload(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def prompt(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    monkeypatch.setattr(
        mcp_prompts,
        "recommend_budget_targets",
        lambda period_start, period_end, savings_target=None, savings_rate=None, db_path=None: {
            "planned_savings": 400.0,
            "category_targets": [{"category_name": "Grocery", "recommended_target": 500.0}],
        },
    )

    mcp_prompts.register_prompts(FakeFastMCP())

    prompt_text = registered["recommend_budget_plan"]("2026-04-28", "2026-05-27", 400.0, None)
    assert "Requested period: 2026-04-28 to 2026-05-27" in prompt_text
    assert "planned_savings" in prompt_text


def test_revise_budget_plan_prompt_includes_revised_payload(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def prompt(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    monkeypatch.setattr(
        mcp_prompts,
        "recommend_budget_targets",
        lambda period_start, period_end, savings_target=None, savings_rate=None, db_path=None: {
            "planned_savings": 400.0,
            "category_targets": [{"category_name": "Grocery", "recommended_target": 500.0}],
        },
    )
    monkeypatch.setattr(
        mcp_prompts,
        "revise_budget_recommendation",
        lambda current_recommendation, user_comment, db_path=None: {
            "planned_savings": 500.0,
            "revision_context": {"protected_categories": ["Bills"]},
        },
    )

    mcp_prompts.register_prompts(FakeFastMCP())

    prompt_text = registered["revise_budget_plan"]({"planned_savings": 400.0}, "raise dine")
    assert "User feedback: raise dine" in prompt_text
    assert "protected_categories" in prompt_text


def test_finalize_budget_plan_prompt_includes_mapped_create_payload(monkeypatch):
    registered = {}

    class FakeFastMCP:
        def prompt(self, **kwargs):
            def decorator(fn):
                registered[kwargs.get("name")] = fn
                return fn
            return decorator

    monkeypatch.setattr(
        mcp_prompts,
        "map_recommendation_to_budget_plan_payload",
        lambda recommendation: {
            "period_start": "2026-05-03",
            "period_end": "2026-06-01",
            "targets": [{"category_name": "Savings", "target_amount": 500.0}],
            "status": "active",
        },
    )

    mcp_prompts.register_prompts(FakeFastMCP())

    prompt_text = registered["finalize_budget_plan"](
        {"planned_savings": 500.0, "category_targets": []},
        "User approved the revised draft.",
    )
    assert "User approved the revised draft." in prompt_text
    assert "Mapped create_budget_plan payload" in prompt_text
    assert "create_budget_plan shape first" in prompt_text
