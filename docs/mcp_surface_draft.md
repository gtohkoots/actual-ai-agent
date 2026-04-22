# MCP Surface Draft

This document sketches the initial MCP contract for the future finance planner.

The purpose of MCP in this system is to expose planner capabilities in a clean and structured way so the agent can work with:

- resources
- tools
- prompts

## MCP Role In This Project

MCP is not the planner engine.

MCP is the interface that presents planner state and planner capabilities to the model runtime.

The deterministic planner engine should compute the truth.
The MCP layer should expose that truth.

## Resource Principles

Use resources for:

- stable, read-mostly state
- planner context that already exists
- things the agent is likely to consult often

Use resources instead of tools when no new computation or mutation is needed.

### Candidate Resources

- `planner://portfolio/current-summary`
- `planner://budget/current-period`
- `planner://goals/active`
- `planner://monitoring/latest`
- `planner://monitoring/history`
- `planner://recommendations/open`
- `planner://preferences/profile`

## Tool Principles

Use tools for:

- targeted computations
- scenario generation
- bounded drilldown
- explicit state updates

Tools should return structured, planner-ready results.

The model should not need to reconstruct finance logic from raw transactions.

### Candidate Read Tools

- `get_portfolio_summary(period, scope?)`
- `get_account_breakdown(period, account_filter?)`
- `get_category_spend(period, category_filter?)`
- `get_income_breakdown(period)`
- `get_transaction_slice(filters)`
- `get_budget_status(period)`
- `get_category_budget_status(period, category)`
- `get_budget_risk_report(period)`
- `forecast_month_end(period)`
- `forecast_category_outcome(period, category)`
- `get_spending_drift(period, baseline?)`
- `estimate_safe_to_spend(period, constraints?)`
- `get_goal_progress(goal_id)`
- `list_goals()`
- `forecast_goal_completion(goal_id, contribution_plan?)`
- `generate_budget_adjustment_options(period)`
- `compare_budget_scenarios(base_scenario, candidate_scenarios)`
- `generate_daily_brief(date?)`

### Candidate Write Tools

These should be approval-gated.

- `create_budget_plan(period, targets)`
- `update_budget_target(period, category, amount)`
- `bulk_rebalance_budget(period, adjustments)`
- `create_goal(name, target_amount, target_date, priority)`
- `update_goal(goal_id, fields)`
- `mark_exception(period_or_category, reason)`
- `acknowledge_recommendation(recommendation_id)`
- `dismiss_recommendation(recommendation_id, reason)`

### Candidate Scenario Tools

- `create_temporary_scenario(name, adjustments)`
- `compare_budget_scenarios(base_scenario, candidate_scenarios)`

These are useful because they let the agent help the user evaluate tradeoffs without mutating persistent planner state.

## Prompt Principles

Use prompts for:

- guided workflows
- recurring user intents
- opinionated entrypoints from the UI

### Candidate Prompts

- `review_current_month`
- `rebalance_budget`
- `plan_large_purchase`
- `create_savings_goal`

## Safety Policy

Read tools:

- safe by default
- can be called without approval

Write tools:

- must be explicit
- should be surfaced in the UI as pending proposals
- require a user confirmation step before execution

## Recommended v1 MCP Scope

Start small.

### v1 Resources

- `planner://portfolio/current-summary`
- `planner://budget/current-period`
- `planner://monitoring/latest`

### v1 Tools

- `get_portfolio_summary`
- `get_budget_status`
- `get_category_budget_status`
- `forecast_month_end`
- `get_spending_drift`

### v1 Prompts

- `review_current_month`
- `plan_large_purchase`

## Why This Surface Works

This split teaches the right abstraction boundaries:

- resources answer "what state already exists?"
- tools answer "what should be computed or changed?"
- prompts answer "what recurring workflow is the user trying to start?"

That is the main mental model to keep when building MCP into this project.
