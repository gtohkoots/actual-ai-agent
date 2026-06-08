from __future__ import annotations

import os
import sqlite3
from typing import Optional


DEFAULT_INVESTMENTS_DB_PATH = "finance_investments.sqlite"


def get_investments_db_path(db_path: Optional[str] = None) -> str:
    return db_path or os.getenv("FINANCE_INVESTMENTS_DB_PATH", DEFAULT_INVESTMENTS_DB_PATH)


def get_investments_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(get_investments_db_path(db_path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def init_investments_db(db_path: Optional[str] = None) -> None:
    with get_investments_connection(db_path) as conn:
        conn.commit()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_imports (
            import_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            filename TEXT,
            as_of_date TEXT,
            imported_at TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'completed'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_accounts (
            account_id TEXT PRIMARY KEY,
            broker TEXT NOT NULL,
            external_account_id TEXT NOT NULL,
            account_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(broker, external_account_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_positions (
            position_id INTEGER PRIMARY KEY AUTOINCREMENT,
            import_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            description TEXT,
            quantity REAL,
            last_price REAL,
            last_price_change REAL,
            current_value REAL NOT NULL,
            day_gain_loss_amount REAL,
            day_gain_loss_percent REAL,
            total_gain_loss_amount REAL,
            total_gain_loss_percent REAL,
            percent_of_account REAL,
            cost_basis_total REAL,
            average_cost_basis REAL,
            fidelity_type TEXT,
            is_cash_like INTEGER NOT NULL DEFAULT 0,
            imported_at TEXT NOT NULL,
            FOREIGN KEY(import_id) REFERENCES investment_imports(import_id),
            FOREIGN KEY(account_id) REFERENCES investment_accounts(account_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_imports_imported_at
        ON investment_imports(imported_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_positions_import_account
        ON investment_positions(import_id, account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_positions_symbol
        ON investment_positions(symbol)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_fund_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            fund_symbol TEXT NOT NULL,
            provider TEXT NOT NULL,
            as_of_date TEXT,
            fetched_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            error_message TEXT,
            raw_payload_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_fund_holdings (
            holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            fund_symbol TEXT NOT NULL,
            constituent_symbol TEXT NOT NULL,
            constituent_name TEXT,
            weight_percent REAL NOT NULL,
            asset_type TEXT,
            sector TEXT,
            FOREIGN KEY(snapshot_id) REFERENCES investment_fund_snapshots(snapshot_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_fund_snapshots_symbol_provider
        ON investment_fund_snapshots(fund_symbol, provider, fetched_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_fund_holdings_snapshot
        ON investment_fund_holdings(snapshot_id, weight_percent DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_industry_classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            industry TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            rationale TEXT,
            provider TEXT NOT NULL,
            classified_at TEXT NOT NULL,
            UNIQUE(symbol, name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_industry_exposure_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            source_import_id TEXT,
            generated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            provider TEXT NOT NULL,
            raw_payload_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS investment_industry_exposures (
            exposure_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            sector TEXT NOT NULL,
            industry TEXT NOT NULL,
            exposure_value REAL NOT NULL,
            percent_of_portfolio REAL NOT NULL,
            company_count INTEGER NOT NULL DEFAULT 0,
            top_companies_json TEXT,
            FOREIGN KEY(snapshot_id) REFERENCES investment_industry_exposure_snapshots(snapshot_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_industry_classifications_symbol
        ON investment_industry_classifications(symbol, name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_investment_industry_exposure_snapshots_generated
        ON investment_industry_exposure_snapshots(generated_at DESC)
        """
    )
    conn.commit()
