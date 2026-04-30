# Planner Agent

This folder contains the first AI-oriented agent layer for the finance planner system.

At the current stage, the planner agent is intentionally small and development-focused. It is not yet embedded into the frontend app as the primary assistant experience.

## What It Does Right Now

The current planner agent supports a minimal budget review flow.

It:

- reads planner state through a real FastMCP client
- calls MCP tools when it needs historical ledger analysis
- calls MCP tools when it needs a savings-aware budget recommendation
- packages that MCP context into a planner prompt payload
- uses an LLM to synthesize the summary, highlights, and next action
- falls back to deterministic output when no OpenAI key is configured

It does not yet:

- forecast month-end outcomes
- generate persistent recommendations
- update budgets automatically
- handle approval workflows
- power the frontend directly

## Files

- [planner_state.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/agents/planner_state.py:1)
  - defines the minimal state object carried through one agent run
- [mcp_client.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/agents/mcp_client.py:1)
  - provides a tiny adapter for reading planner resources and calling planner tools through the FastMCP runtime
- [prompts.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/agents/prompts.py:1)
  - defines the planner system prompt and the MCP-derived prompt context shape
- [llm.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/agents/llm.py:1)
  - isolates model invocation and fallback behavior
- [planner_agent.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/agents/planner_agent.py:1)
  - contains the MCP-backed orchestration flow for the planner agent
- [run_planner_agent.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/agents/run_planner_agent.py:1)
  - provides a simple CLI/dev entrypoint

## How It Works

The current flow is:

1. Accept a user message.
2. Build a FastMCP runtime and client session.
3. Read the active budget plan through MCP.
4. Read the current budget status through MCP.
5. When the user asks for historical review, call the relevant MCP ledger-analysis tools.
6. Build a planner prompt context from the MCP resources and tool results.
7. Ask the LLM to produce:
   - a summary
   - highlights
   - a next action
8. Return the response together with the MCP context it used.

This is the first real `agent -> MCP runtime -> planner resources -> LLM -> response` path in the repo.

## Resource Usage

The current agent reads these planner resources through the MCP runtime:

- `planner://budget/active-plan`
- `planner://budget/current-status`

It can also call these MCP tools for historical review:

- `get_portfolio_summary`
- `get_category_spend`
- `get_account_breakdown`
- `get_spending_drift`

For budget recommendation flows it can call:

- `recommend_budget_targets`

These resources are implemented in:

- [backend/mcp/resources.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend/mcp/resources.py:1)

## How To Run It

From the repo root:

```bash
./.venv/bin/python -m backend.agents.run_planner_agent "Review my budget"
```

Optional planner DB override:

```bash
./.venv/bin/python -m backend.agents.run_planner_agent "Review my budget" --db-path /path/to/planner.sqlite
```

## Why This Layer Exists

This folder exists to separate:

- planner business logic in `backend/services`
- MCP capability exposure in `backend/mcp`
- agent reasoning/orchestration in `backend/agents`

That separation keeps the architecture easier to evolve and reason about.

## Next Expected Evolution

Likely future additions in this folder:

- richer agent flows
- approval-aware actions
- scenario analysis
- forecasting-aware planner responses
- eventual app-facing planner assistant integration
