import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.langchain_runner import get_weekly_data_tool
from backend.services.insights import get_week_rollups, save_daily_snapshot
from backend.utils.db import get_transactions_in_date_range


def test_get_transactions_in_date_range_converts_amounts_and_exposes_aliases():
    df = get_transactions_in_date_range("2026-03-16", "2026-03-16", debug=False)

    assert "category" in df.columns
    assert "account" in df.columns
    assert df["amount"].dtype.kind == "f"
    assert df.iloc[0]["amount"] == -178.73


def test_week_rollups_use_human_category_names():
    payload = get_week_rollups("2026-03-16", "2026-03-22")
    categories = {item["category"] for item in payload["by_category"]}

    assert "nan" not in categories
    assert "(uncategorized)" not in categories
    assert "Grocery" in categories
    assert "Dine" in categories


def test_get_weekly_data_tool_returns_dollar_amounts_and_category_fields():
    payload = json.loads(
        get_weekly_data_tool.invoke({"start_date": "2026-03-16", "end_date": "2026-03-16"})
    )

    assert payload
    assert payload[0]["amount"] == -178.73
    assert "category" in payload[0]
    assert "account" in payload[0]


def test_filter_ignored_payment_is_noop():
    df = pd.DataFrame(
        {
            "category_name": ["Ignored - Expense", "Ignored - Income", "Food"],
            "amount": [-10.0, 10.0, -5.0],
        }
    )

    from backend.services.filters import filter_ignored_payment

    filtered = filter_ignored_payment(df)

    assert filtered["category_name"].tolist() == ["Ignored - Expense", "Ignored - Income", "Food"]


def test_save_daily_snapshot_returns_empty_message_for_dates_without_transactions():
    result = save_daily_snapshot("1999-01-01", 200.0)

    assert result == "1999-01-01 无交易数据，未生成快照。"
