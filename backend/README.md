# Backend

This folder contains the Python backend for the finance agent.

## Layout

- `app.py`: FastAPI app exposing account, dashboard, analysis chat, planner chat, and document endpoints
- `agents/`: analysis, planner, and shared orchestration code
- `services/`: API-facing wrappers, budget services, retrieval, filters, and document ingestion
- `utils/`: Actual database access helpers
- `tests/`: backend regression tests
- `requirements.txt`: backend Python dependencies

## Run

From the repo root, start the API server for the React frontend:

```bash
./.venv/bin/uvicorn backend.app:app --reload
```

## Notes

- The backend is intentionally separated from the React frontend so the chat API can evolve without coupling it to the CLI layout.
- Shared state should be request-scoped for chat requests.
- The API currently exposes `/api/health`, `/api/accounts`, `/api/dashboard`, `/api/analysis/chat`, `/api/planner/chat`, `/api/planner/overview`, and document search/rebuild endpoints.
