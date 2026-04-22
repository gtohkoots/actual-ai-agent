import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.planner_db import get_planner_connection, init_planner_db


def test_init_planner_db_creates_minimal_tables(tmp_path):
    db_path = tmp_path / "planner.sqlite"

    init_planner_db(str(db_path))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

    table_names = {row[0] for row in rows}
    assert {"budget_plans", "budget_targets", "goals", "planner_preferences"} <= table_names


def test_get_planner_connection_uses_row_factory_and_bootstraps_schema(tmp_path):
    db_path = tmp_path / "planner.sqlite"

    with get_planner_connection(str(db_path)) as conn:
        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'budget_plans'
            """
        ).fetchone()

    assert isinstance(row, sqlite3.Row)
    assert row["name"] == "budget_plans"


def test_init_planner_db_is_idempotent(tmp_path):
    db_path = tmp_path / "planner.sqlite"

    init_planner_db(str(db_path))
    init_planner_db(str(db_path))

    with sqlite3.connect(db_path) as conn:
        indexes = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
            """
        ).fetchall()

    index_names = {row[0] for row in indexes}
    assert "idx_budget_plans_status_period" in index_names
    assert "idx_budget_targets_plan_category" in index_names
    assert "idx_goals_status_priority" in index_names
