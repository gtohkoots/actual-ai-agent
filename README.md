# Finance Agent

A small personal finance assistant built on top of:

- an Actual Budget SQLite database
- Python data processing with `pandas`
- a LangChain tool-calling agent
- OpenAI chat models for report generation
- a React frontend prototype for card-spending exploration and AI chat

The app reads transactions from your Actual database, computes weekly and daily summaries, and can generate Chinese-language finance reports from those facts.

## What The App Does

The agent can:

- set a time window for analysis
- load transactions for a date range
- compute weekly rollups such as income, expense, net cash flow, top categories, and top payees
- compare one week to the previous week
- save weekly reports as Markdown
- save daily snapshots as JSON

Generated files are stored in:

- [daily_snapshots](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/daily_snapshots)
- [weekly_snapshots](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/weekly_snapshots)
- [weekly_reports](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/weekly_reports)

## Project Layout

- [langchain_runner.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/langchain_runner.py): interactive CLI app and agent tools
- [utils/db.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/utils/db.py): SQLite access layer for Actual transactions
- [services/insights.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/services/insights.py): rollups, comparisons, anomaly helpers, and snapshot logic
- [services/documents.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/services/documents.py): historical artifact ingestion and retrieval helpers
- [services/filters.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/services/filters.py): filters for ignored internal payment rows
- [tests/test_phase1_finance_agent.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/tests/test_phase1_finance_agent.py): regression tests for the data normalization fixes
- [tests/test_phase2_documents.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/tests/test_phase2_documents.py): regression tests for document ingestion and retrieval
- [frontend](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend): React frontend with mocked card, spending, and AI chat data

## Requirements

- Python 3.10+
- an Actual Budget SQLite database file
- an OpenAI API key

## Setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
ACTUAL_DB_PATH=/absolute/path/to/your/db.sqlite
OPENAI_API_KEY=your_openai_api_key
```

Optional:

```env
AMEX_ACCT_ID=your_account_id
```

`AMEX_ACCT_ID` is present in the sample environment but is not currently used by the app.

## Run The App

Start the interactive CLI:

```bash
./.venv/bin/python langchain_runner.py
```

You will see example prompts such as:

- `生成一份本周的详细周报并保存`
- `将时间设为 2025-08-01 到 2025-08-07，然后给我详细报告`
- `这周和上周相比，支出变化如何？`

Type `exit` or `quit` to leave the app.

## Run The Frontend

Start the React frontend with mocked data:

```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL shown in the terminal.

## How It Works

1. The app reads transactions from the Actual SQLite database.
2. The agent calls internal tools to fetch or summarize data.
3. The LLM turns those structured results into a Chinese-language financial report.
4. If requested, the app writes snapshots or reports to disk.

The current system is a tool-using finance agent rather than a full RAG system. It primarily answers from live ledger data and computed summaries.

Phase 2 adds a lightweight document layer by converting saved snapshots and reports into a local SQLite document store. This gives the agent a historical corpus it can search later without recomputing everything from the ledger.

## Available Agent Tools

Inside the CLI, the agent can call these tools:

- `update_time_window_tool`: set the active date range
- `get_weekly_data_tool`: return raw transactions for a date range
- `get_week_rollups_tool`: return weekly rollups and save a JSON snapshot
- `compare_to_last_week_tool`: compare a week against the prior week
- `save_weekly_report_tool`: save a Markdown report
- `save_daily_snapshot_tool`: save a one-day JSON snapshot
- `refresh_artifact_documents_tool`: rebuild the historical document store from saved artifacts
- `search_artifact_documents_tool`: search the saved document corpus by keyword, type, and date range
- `search_past_weeks_by_category_tool`: find past weekly snapshots for a specific category
- `find_similar_spending_weeks_tool`: find historically similar weekly spending patterns
- `get_recent_anomalies_tool`: retrieve recent large-expense anomalies from historical snapshots
- `search_reports_tool`: search prior weekly reports for themes, advice, or narrative context

## Historical Documents

Phase 2 turns these artifact folders into documents:

- `daily_snapshots/*.json`
- `weekly_snapshots/*.json`
- `weekly_reports/*.md`

The normalized documents are stored in a local SQLite file:

- `finance_documents.sqlite`

Each document stores:

- document type
- source file path
- title and content
- date range
- income, expense, and net cash flow when available
- categories and payees when available
- structured metadata as JSON

You can rebuild the document store manually by running the app and asking it to refresh historical documents, or by calling the Python helper in [services/documents.py](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/services/documents.py).

## Explicit Retrieval

Phase 3 adds explicit retrieval helpers on top of the document store so the agent can ask focused historical questions instead of relying on one generic search.

Examples:

- "Find past weeks with Grocery spending"
- "Show weeks similar to 2026-03-16 through 2026-03-22"
- "Get recent anomalies for Costco"
- "Search past reports for advice about negative cash flow"

## Outputs

### Weekly Snapshot

Saved as JSON in `weekly_snapshots/` and includes:

- time window
- total income
- total expense
- net cash flow
- top categories
- top payees
- large expenses
- income payee distribution

### Weekly Report

Saved as Markdown in `weekly_reports/` and typically includes:

- facts summary
- income section
- expense section
- net cash flow
- week-over-week comparison
- unusual or large expenses
- suggestions and budget commentary

### Daily Snapshot

Saved as JSON in `daily_snapshots/` and includes:

- total income
- total expense
- category summary
- notes for large expenses

## Running Tests

Run the regression tests with:

```bash
./.venv/bin/pytest -q
```

## Notes And Limitations

- The app currently assumes a Chinese-language reporting workflow.
- The agent uses LangChain's older import path for `ChatOpenAI`, so you may see deprecation warnings during tests or execution.
- The app depends on the schema of an Actual Budget SQLite database.
- This project currently does not implement a true historical retrieval or vector-based RAG layer.

## Troubleshooting

### `ACTUAL_DB_PATH is not set`

Make sure your `.env` file exists and points to a valid SQLite database file.

### OpenAI authentication errors

Make sure `OPENAI_API_KEY` is set in your environment or `.env` file.

### `ModuleNotFoundError`

Make sure dependencies were installed into the same virtual environment you are using to run the app.

### No data returned

Check that:

- the date range contains transactions
- `ACTUAL_DB_PATH` points to the correct database
- your Actual database schema matches what the code expects
