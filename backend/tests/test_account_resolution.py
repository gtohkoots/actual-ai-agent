import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.utils.db import get_connection, get_transactions_in_date_range, resolve_account_reference


def _account_pid(name: str) -> str:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM accounts WHERE name = ?", (name,)).fetchone()
    assert row is not None, f"Missing account named {name!r} in test DB"
    return row["id"]


def test_resolve_account_reference_prefers_pid_over_name():
    pid = _account_pid("Amex Gold Card")

    resolved = resolve_account_reference(account_pid=pid, account_name="Not The Right Account")

    assert resolved is not None
    assert resolved["account_pid"] == pid
    assert resolved["account_name"] == "Amex Gold Card"


def test_get_transactions_in_date_range_uses_pid_even_if_name_mismatches():
    pid = _account_pid("Amex Gold Card")

    df = get_transactions_in_date_range(
        "2026-03-16",
        "2026-03-20",
        account_pid=pid,
        account_name="Definitely Wrong",
        debug=False,
    )

    assert not df.empty
    assert set(df["account_pid"].astype(str)) == {pid}
    assert set(df["account"].astype(str)) == {"Amex Gold Card"}


def test_get_transactions_in_date_range_can_fallback_to_exact_name():
    df = get_transactions_in_date_range(
        "2026-03-16",
        "2026-03-20",
        account_name="Amex Gold Card",
        debug=False,
    )

    assert not df.empty
    assert set(df["account"].astype(str)) == {"Amex Gold Card"}

