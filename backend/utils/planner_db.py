from __future__ import annotations

import os
import sqlite3
from typing import Optional


DEFAULT_PLANNER_DB_PATH = "finance_planner.sqlite"


def get_planner_db_path(db_path: Optional[str] = None) -> str:
    """Resolve the planner DB path from an explicit argument or environment."""
    return db_path or os.getenv("FINANCE_PLANNER_DB_PATH", DEFAULT_PLANNER_DB_PATH)


def get_planner_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a planner DB connection and ensure the minimal schema exists."""
    conn = sqlite3.connect(get_planner_db_path(db_path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def init_planner_db(db_path: Optional[str] = None) -> None:
    """Initialize the planner DB schema eagerly for scripts or tests."""
    with get_planner_connection(db_path) as conn:
        conn.commit()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the minimal planner tables and indexes if they do not exist."""
    # Keep planner state in its own SQLite schema so budgets/goals can evolve
    # independently from the ledger DB and the chat/document stores.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS budget_plans (
            plan_id TEXT PRIMARY KEY,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS budget_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT NOT NULL,
            category_name TEXT NOT NULL,
            target_amount REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(plan_id) REFERENCES budget_plans(plan_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS goals (
            goal_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            goal_type TEXT NOT NULL,
            target_amount REAL NOT NULL,
            target_date TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            priority INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS planner_preferences (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_budget_plans_status_period
        ON budget_plans(status, period_start, period_end)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_budget_targets_plan_category
        ON budget_targets(plan_id, category_name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_goals_status_priority
        ON goals(status, priority)
        """
    )
    conn.commit()
