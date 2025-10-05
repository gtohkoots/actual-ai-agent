import os
import sqlite3
from typing import Iterable, Optional
from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 文件

import pandas as pd

DEFAULT_DB_PATH = os.getenv("ACTUAL_DB_PATH")  # e.g., /path/to/budget.sqlite


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    if not path:
        raise RuntimeError("ACTUAL_DB_PATH is not set. Please set it in environment or pass db_path.")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _to_yyyymmdd_int(date_str: str) -> int:
    return int(pd.to_datetime(date_str).strftime("%Y%m%d"))


def get_transactions_in_date_range(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    join_names: bool = True,
    dollars: bool = True,
    columns: Optional[Iterable[str]] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Load transactions within [start_date, end_date] (inclusive).
    - Converts Actual's integer date (YYYYMMDD) to pandas datetime64[ns]
    - Optionally joins category/account names
    - Returns amounts in USD (float) if dollars=True
    - Default columns: date, amount, payee(imported_description), category, account, notes
    """
    s_int = _to_yyyymmdd_int(start_date)
    e_int = _to_yyyymmdd_int(end_date)

    if debug:
        print(f"Loading transactions from {start_date} to {end_date} ({s_int} to {e_int})")

    with get_connection(db_path) as conn:
        tx = pd.read_sql_query(
            "SELECT * FROM transactions WHERE date BETWEEN ? AND ?",
            conn,
            params=(s_int, e_int),
        )
        if debug:
            print("transactions sample:\n", tx.head())

        if join_names:
            categories = pd.read_sql_query(
                "SELECT id AS category_id, name AS category_name FROM categories",
                conn,
            )
            accounts = pd.read_sql_query(
                "SELECT id AS account_pid, name AS account_name FROM accounts",
                conn,
            )
        else:
            categories = accounts = None

    # date conversion
    tx["date"] = pd.to_datetime(tx["date"].astype(str), format="%Y%m%d")

    # payee from imported_description
    if "imported_description" in tx.columns:
        tx["payee"] = tx["imported_description"]
    elif "payee" not in tx.columns:
        tx["payee"] = None

    print(f"Loaded {len(tx)} transactions, {len(categories)} categories, {len(accounts)} accounts")

    # join human-readable names
    if join_names and categories is not None:
        tx = tx.merge(categories, left_on="category", right_on="category_id", how="left")
        tx = tx.merge(accounts, left_on="acct", right_on="account_pid", how="left")

    # choose columns
    base_cols = ["date", "amount", "payee", "category_name", "account_name", "account_pid", "notes"]
    cols = list(columns) if columns is not None else [c for c in base_cols if c in tx.columns]
    df = tx[cols].copy()

    df = df.sort_values("date").reset_index(drop=True)

    if debug:
        print("final flattened DataFrame columns:", df.columns.tolist())
        print("total rows returned:", len(df))

    return df


