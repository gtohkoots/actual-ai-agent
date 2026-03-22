# Frontend Prototype

This folder contains the React frontend for the finance agent UI. The dashboard still uses mocked financial data for now, but the chatbot is wired to the backend API.

## Included

- [index.html](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/index.html): Vite entry HTML
- [package.json](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/package.json): React app scripts and dependencies
- [src/App.jsx](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/App.jsx): main React app
- [src/components/ChatPanel.jsx](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/components/ChatPanel.jsx): interactive AI chatbot interface
- [src/chat/api.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/chat/api.js): chat API client for the backend endpoint
- [src/chat/mockResponder.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/chat/mockResponder.js): welcome-message and mock fallback helpers
- [src/mockData.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/mockData.js): mocked finance/card/chat data
- [src/styles.css](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/styles.css): visual system and responsive layout

## What This Prototype Demonstrates

- card-first dashboard navigation
- month-level spending overview
- contextual AI chat panel
- Markdown-rendered assistant replies
- source cards and action chips for follow-up interaction
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

This is still using mocked dashboard data, but the chatbot now sends real requests to `POST /api/chat`.

## Recommended Next Steps

1. Connect the dashboard cards and tables to backend data.
2. Add streaming responses for the chatbot.
3. Split the UI into more reusable components as the product solidifies.
