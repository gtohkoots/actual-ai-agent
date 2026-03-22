# Frontend Prototype

This folder now contains a React frontend for the finance agent UI, using mocked data so we can iterate on product flow before wiring the backend.

## Included

- [index.html](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/index.html): Vite entry HTML
- [package.json](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/package.json): React app scripts and dependencies
- [src/App.jsx](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/App.jsx): main React app
- [src/mockData.js](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/mockData.js): mocked finance/card/chat data
- [src/styles.css](/Users/ketia/Documents/Actual/My-Finances-b5b9544/finance-agent/frontend/src/styles.css): visual system and responsive layout

## What This Prototype Demonstrates

- card-first dashboard navigation
- month-level spending overview
- contextual AI chat panel
- category and merchant breakdowns
- recent transaction table
- quick prompts that adapt to the selected card

## Open It

From the repo root:

```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL shown in the terminal.

Generated folders:

- `node_modules/`
- `dist/`

These are intentionally ignored by Git and should not be committed.

## Current Status

This is still using mocked data, but it is now structured as a React app and ready for component-level iteration and backend API integration.

## Recommended Next Steps

1. Keep this layout direction if it feels right.
2. Add a small backend API layer for cards, transactions, dashboard summaries, and chat.
3. Replace the mock data with live data from the finance agent.
4. Split the UI into reusable components as the product solidifies.
