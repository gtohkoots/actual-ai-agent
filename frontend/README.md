# Frontend Prototype

This folder contains the React frontend for the finance agent UI. The dashboard now loads live account and summary data from the backend, and the chatbot is wired to the backend API.

## Included

- [index.html](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/index.html): Vite entry HTML
- [package.json](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/package.json): React app scripts and dependencies
- [src/App.jsx](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/App.jsx): main React app
- [src/api/backend.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/api/backend.js): shared backend URL and message normalization helpers
- [src/components/ChatPanel.jsx](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/components/ChatPanel.jsx): interactive AI chatbot interface
- [src/chat/api.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/chat/api.js): chat API client for the backend endpoint
- [src/planner/api.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/planner/api.js): planner chat API client for the planner assistant endpoint
- [src/chat/mockResponder.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/chat/mockResponder.js): welcome-message and mock fallback helpers
- [src/mockData.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/mockData.js): legacy mocked finance/card/chat data used as a design fallback
- [src/styles.css](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/styles.css): visual system and responsive layout

## What This Prototype Demonstrates

- card-first dashboard navigation
- month-level spending overview
- contextual AI chat panel
- Markdown-rendered assistant replies
- source cards and action chips for follow-up interaction
- selected-card context includes `account_pid` so the backend loads the right account deterministically
- category and merchant breakdowns
- recent transaction table
- quick prompts that adapt to the selected card

## Open It

From the repo root:

```bash
./.venv/bin/uvicorn backend.app:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL shown in the terminal.

If your backend runs somewhere other than `http://127.0.0.1:8000`, set `VITE_BACKEND_URL` before starting Vite.

Generated folders:

- `node_modules/`
- `dist/`

These are intentionally ignored by Git and should not be committed.

## Current Status

The UI now hydrates accounts and dashboard summaries from `GET /api/accounts` and `GET /api/dashboard`, while the chatbot continues to send real requests to `POST /api/chat`.

## Recommended Next Steps

1. Connect the transactions table to its own backend endpoint so it can be paginated and filtered.
2. Wire `ChatPanel` into planner mode using the planner chat client.
3. Add streaming responses for the chatbot.
4. Split the UI into more reusable components as the product solidifies.
