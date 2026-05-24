# Finance Agent

A small personal finance assistant built on top of:

- an Actual Budget SQLite database
- Python data processing with `pandas`
- OpenAI chat models for grounded analysis and planning
- a React frontend for card-spending exploration and AI chat
- a dedicated Python backend package under `backend/`

The app reads transactions from your Actual database, computes financial summaries, and powers both analysis chat and budgeting workflows from those facts.

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

- [backend](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/backend): Python backend package, CLI, services, and tests
- [frontend](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend): React frontend with mocked card/spending data and a live AI chat endpoint

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

3. Install backend dependencies:

```bash
pip install -r backend/requirements.txt
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

## Run The Backend

Start the API server for the frontend assistant:

```bash
./.venv/bin/uvicorn backend.app:app --reload
```

The frontend can call:

- `GET /api/accounts`
- `GET /api/dashboard`
- `GET /api/health`
- `POST /api/analysis/chat`
- `POST /api/planner/chat`
- `GET /api/planner/overview`
- `POST /api/documents/rebuild`
- `GET /api/documents/search`

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
2. The backend computes grounded summaries and budgeting context.
3. Analysis and planner assistants turn those structured results into user-facing responses.
4. Historical artifacts can be rebuilt into a document layer for retrieval.

The current system is a grounded finance assistant rather than a free-form autonomous agent. It primarily answers from live ledger data, computed summaries, and planner workflows.

The document layer converts saved snapshots and reports into a local SQLite document store so the assistants can search historical artifacts later without recomputing everything from the ledger.

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

- The app currently includes both an analysis assistant and a planner assistant.
- The backend uses `langchain-openai` for model access in the assistant flows.
- The app depends on the schema of an Actual Budget SQLite database.
- This project currently does not implement a true historical retrieval or vector-based RAG layer.
- Future hardening idea: replace frontend-visible account PIDs with opaque account keys and add backend authorization checks once the app is multi-user.

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
