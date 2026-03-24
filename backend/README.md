# Backend

This folder contains the Python backend for the finance agent.

## Layout

- `langchain_runner.py`: CLI entrypoint and tool orchestration
- `app.py`: FastAPI app exposing account, dashboard, chat, and document endpoints
- `services/`: rollups, retrieval, filters, and document ingestion
- `utils/`: Actual database access helpers
- `tests/`: backend regression tests
- `requirements.txt`: backend Python dependencies

## Run

From the repo root:

```bash
./.venv/bin/python -m backend.langchain_runner
```

Run the API server for the React frontend:

```bash
./.venv/bin/uvicorn backend.app:app --reload
```

## Notes

- The backend is intentionally separated from the React frontend so the chat API can evolve without coupling it to the CLI layout.
- Shared state should be request-scoped for chat requests.
- The API currently exposes `/api/health`, `/api/accounts`, `/api/dashboard`, `/api/chat`, and document search/rebuild endpoints.
