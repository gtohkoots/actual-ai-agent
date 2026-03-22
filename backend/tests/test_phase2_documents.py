import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.documents import (
    find_similar_spending_weeks,
    get_recent_anomalies,
    rebuild_document_store,
    search_documents,
    search_past_weeks_by_category,
    search_reports,
)


def test_rebuild_document_store_indexes_existing_artifacts(tmp_path):
    daily_dir = tmp_path / "daily_snapshots"
    weekly_dir = tmp_path / "weekly_snapshots"
    report_dir = tmp_path / "weekly_reports"
    daily_dir.mkdir()
    weekly_dir.mkdir()
    report_dir.mkdir()

    (daily_dir / "2026-03-20.json").write_text(
        json.dumps(
            {
                "date": "2026-03-20",
                "total_income": 100.0,
                "total_expense": 40.0,
                "categories": {"income": {"Salary": 100.0}, "expense": {"Food": 40.0}},
                "notes": "Lunch and coffee",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (weekly_dir / "week_2026-03-16_to_2026-03-22.json").write_text(
        json.dumps(
            {
                "window": {"start": "2026-03-16", "end": "2026-03-22"},
                "summary": {"total_income": 1000.0, "total_expense": 800.0, "net_cashflow": 200.0},
                "by_category": [{"category": "Food", "amount": 300.0}],
                "top_payees": [{"payee": "Costco", "amount": 180.0}],
                "big_expenses": [{"date": "2026-03-18", "payee": "Costco", "category": "Food", "amount": 180.0}],
                "income_payee_distribution": [{"payee": "Employer", "amount": 1000.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (report_dir / "weekly_report_2026-03-22.md").write_text(
        "# Weekly Report\n\nCostco spending increased this week.",
        encoding="utf-8",
    )

    counts = rebuild_document_store(str(tmp_path), db_path=str(tmp_path / "documents.sqlite"))

    assert counts == {
        "daily_snapshot": 1,
        "weekly_snapshot": 1,
        "weekly_report": 1,
        "total": 3,
    }


def test_search_documents_filters_by_query_type_and_date(tmp_path):
    daily_dir = tmp_path / "daily_snapshots"
    weekly_dir = tmp_path / "weekly_snapshots"
    report_dir = tmp_path / "weekly_reports"
    daily_dir.mkdir()
    weekly_dir.mkdir()
    report_dir.mkdir()

    (daily_dir / "2026-03-20.json").write_text(
        json.dumps(
            {
                "date": "2026-03-20",
                "total_income": 0.0,
                "total_expense": 25.0,
                "categories": {"income": {}, "expense": {"Coffee": 25.0}},
                "notes": "Cafe visit",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (weekly_dir / "week_2026-03-16_to_2026-03-22.json").write_text(
        json.dumps(
            {
                "window": {"start": "2026-03-16", "end": "2026-03-22"},
                "summary": {"total_income": 1000.0, "total_expense": 800.0, "net_cashflow": 200.0},
                "by_category": [{"category": "Grocery", "amount": 260.53}],
                "top_payees": [{"payee": "Costco", "amount": 178.73}],
                "big_expenses": [],
                "income_payee_distribution": [{"payee": "Employer", "amount": 1000.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (report_dir / "weekly_report_2026-03-22.md").write_text(
        "# Weekly Report\n\nCostco spending increased this week.",
        encoding="utf-8",
    )

    db_path = str(tmp_path / "documents.sqlite")
    rebuild_document_store(str(tmp_path), db_path=db_path)

    results = search_documents(query="Costco", doc_type="weekly_snapshot", limit=5, db_path=db_path)

    assert len(results) == 1
    assert results[0]["doc_type"] == "weekly_snapshot"
    assert results[0]["payees"] == ["Costco"]

    date_filtered = search_documents(
        start_date="2026-03-22",
        end_date="2026-03-22",
        limit=5,
        db_path=db_path,
    )
    assert {item["doc_type"] for item in date_filtered} == {"weekly_snapshot", "weekly_report"}


def test_rebuild_document_store_accepts_legacy_empty_daily_snapshot_strings(tmp_path):
    daily_dir = tmp_path / "daily_snapshots"
    weekly_dir = tmp_path / "weekly_snapshots"
    report_dir = tmp_path / "weekly_reports"
    daily_dir.mkdir()
    weekly_dir.mkdir()
    report_dir.mkdir()

    (daily_dir / "2026-03-19.json").write_text(
        json.dumps("2026-03-19 无交易数据，未生成快照。", ensure_ascii=False),
        encoding="utf-8",
    )

    db_path = str(tmp_path / "documents.sqlite")
    counts = rebuild_document_store(str(tmp_path), db_path=db_path)
    results = search_documents(query="无交易数据", limit=5, db_path=db_path)

    assert counts == {"daily_snapshot": 1, "total": 1}
    assert len(results) == 1
    assert results[0]["metadata"]["status"] == "empty"


def test_explicit_retrieval_tools_return_targeted_history(tmp_path):
    daily_dir = tmp_path / "daily_snapshots"
    weekly_dir = tmp_path / "weekly_snapshots"
    report_dir = tmp_path / "weekly_reports"
    daily_dir.mkdir()
    weekly_dir.mkdir()
    report_dir.mkdir()

    weekly_payloads = {
        "week_2026-03-16_to_2026-03-22.json": {
            "window": {"start": "2026-03-16", "end": "2026-03-22"},
            "summary": {"total_income": 1000.0, "total_expense": 800.0, "net_cashflow": 200.0},
            "by_category": [
                {"category": "Grocery", "amount": 260.53},
                {"category": "Food", "amount": 120.0},
            ],
            "top_payees": [{"payee": "Costco", "amount": 178.73}],
            "big_expenses": [{"date": "2026-03-18", "payee": "Costco", "category": "Grocery", "amount": 178.73}],
            "income_payee_distribution": [{"payee": "Employer", "amount": 1000.0}],
        },
        "week_2026-03-09_to_2026-03-15.json": {
            "window": {"start": "2026-03-09", "end": "2026-03-15"},
            "summary": {"total_income": 950.0, "total_expense": 780.0, "net_cashflow": 170.0},
            "by_category": [
                {"category": "Grocery", "amount": 240.0},
                {"category": "Bills", "amount": 130.0},
            ],
            "top_payees": [{"payee": "Trader Joe's", "amount": 150.0}],
            "big_expenses": [{"date": "2026-03-12", "payee": "Progressive", "category": "Bills", "amount": 130.0}],
            "income_payee_distribution": [{"payee": "Employer", "amount": 950.0}],
        },
        "week_2026-03-02_to_2026-03-08.json": {
            "window": {"start": "2026-03-02", "end": "2026-03-08"},
            "summary": {"total_income": 900.0, "total_expense": 400.0, "net_cashflow": 500.0},
            "by_category": [{"category": "Travel", "amount": 150.0}],
            "top_payees": [{"payee": "Delta", "amount": 150.0}],
            "big_expenses": [{"date": "2026-03-05", "payee": "Delta", "category": "Travel", "amount": 150.0}],
            "income_payee_distribution": [{"payee": "Employer", "amount": 900.0}],
        },
    }
    for filename, payload in weekly_payloads.items():
        (weekly_dir / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    (report_dir / "weekly_report_2026-03-22.md").write_text(
        "# Weekly Report\n\nCostco spending increased and grocery expenses were elevated.",
        encoding="utf-8",
    )

    db_path = str(tmp_path / "documents.sqlite")
    rebuild_document_store(str(tmp_path), db_path=db_path)

    category_results = search_past_weeks_by_category("Grocery", limit=5, db_path=db_path)
    assert [item["start_date"] for item in category_results] == ["2026-03-16", "2026-03-09"]
    assert category_results[0]["category_amount"] == 260.53

    similar_weeks = find_similar_spending_weeks("2026-03-16", "2026-03-22", limit=2, db_path=db_path)
    assert similar_weeks[0]["start_date"] == "2026-03-09"
    assert "Grocery" in similar_weeks[0]["shared_categories"]

    anomalies = get_recent_anomalies(payee="Costco", limit=5, db_path=db_path)
    assert len(anomalies) == 1
    assert anomalies[0]["category"] == "Grocery"

    report_matches = search_reports("Costco", limit=5, db_path=db_path)
    assert len(report_matches) == 1
    assert "Costco" in report_matches[0]["excerpt"]
