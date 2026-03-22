from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_DOCUMENT_DB_PATH = os.getenv("FINANCE_DOCUMENT_DB_PATH", "finance_documents.sqlite")


@dataclass
class FinanceDocument:
    doc_id: str
    doc_type: str
    source_path: str
    title: str
    content: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_income: Optional[float]
    total_expense: Optional[float]
    net_cashflow: Optional[float]
    categories: List[str]
    payees: List[str]
    metadata: Dict[str, Any]


def get_document_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DEFAULT_DOCUMENT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_document_store(db_path: Optional[str] = None) -> None:
    with get_document_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS finance_documents (
                doc_id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                total_income REAL,
                total_expense REAL,
                net_cashflow REAL,
                categories_json TEXT NOT NULL,
                payees_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_finance_documents_type ON finance_documents(doc_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_finance_documents_dates ON finance_documents(start_date, end_date)"
        )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_daily_document(path: Path) -> FinanceDocument:
    payload = _read_json(path)
    if isinstance(payload, str):
        return FinanceDocument(
            doc_id=f"daily_snapshot::{path.stem}",
            doc_type="daily_snapshot",
            source_path=str(path),
            title=f"Daily snapshot {path.stem}",
            content=payload,
            start_date=path.stem,
            end_date=path.stem,
            total_income=None,
            total_expense=None,
            net_cashflow=None,
            categories=[],
            payees=[],
            metadata={"message": payload, "status": "empty"},
        )

    categories = payload.get("categories", {})
    expense_categories = list(categories.get("expense", {}).keys())
    income_categories = list(categories.get("income", {}).keys())
    notes = payload.get("notes", "")
    content = "\n".join(
        [
            f"Daily snapshot for {payload.get('date')}",
            f"Total income: {payload.get('total_income', 0.0)}",
            f"Total expense: {payload.get('total_expense', 0.0)}",
            f"Income categories: {', '.join(income_categories) or '(none)'}",
            f"Expense categories: {', '.join(expense_categories) or '(none)'}",
            f"Notes: {notes or '(none)'}",
        ]
    )
    return FinanceDocument(
        doc_id=f"daily_snapshot::{path.stem}",
        doc_type="daily_snapshot",
        source_path=str(path),
        title=f"Daily snapshot {payload.get('date', path.stem)}",
        content=content,
        start_date=payload.get("date"),
        end_date=payload.get("date"),
        total_income=payload.get("total_income"),
        total_expense=payload.get("total_expense"),
        net_cashflow=round(float(payload.get("total_income", 0.0)) - float(payload.get("total_expense", 0.0)), 2),
        categories=income_categories + expense_categories,
        payees=[],
        metadata={
            "notes": notes,
            "categories": categories,
        },
    )


def _extract_weekly_snapshot_document(path: Path) -> FinanceDocument:
    payload = _read_json(path)
    summary = payload.get("summary", {})
    by_category = payload.get("by_category", [])
    top_payees = payload.get("top_payees", [])
    big_expenses = payload.get("big_expenses", [])
    income_payee_distribution = payload.get("income_payee_distribution", [])
    categories = [item.get("category") for item in by_category if item.get("category")]
    payees = [item.get("payee") for item in top_payees if item.get("payee")]
    content = "\n".join(
        [
            f"Weekly snapshot for {payload.get('window', {}).get('start')} to {payload.get('window', {}).get('end')}",
            f"Total income: {summary.get('total_income', 0.0)}",
            f"Total expense: {summary.get('total_expense', 0.0)}",
            f"Net cashflow: {summary.get('net_cashflow', 0.0)}",
            f"Top categories: {', '.join(categories) or '(none)'}",
            f"Top payees: {', '.join(payees) or '(none)'}",
            f"Income payees: {', '.join(item.get('payee', '') for item in income_payee_distribution if item.get('payee')) or '(none)'}",
            f"Big expenses: {', '.join(item.get('payee', '') for item in big_expenses if item.get('payee')) or '(none)'}",
        ]
    )
    return FinanceDocument(
        doc_id=f"weekly_snapshot::{path.stem}",
        doc_type="weekly_snapshot",
        source_path=str(path),
        title=f"Weekly snapshot {payload.get('window', {}).get('start')} to {payload.get('window', {}).get('end')}",
        content=content,
        start_date=payload.get("window", {}).get("start"),
        end_date=payload.get("window", {}).get("end"),
        total_income=summary.get("total_income"),
        total_expense=summary.get("total_expense"),
        net_cashflow=summary.get("net_cashflow"),
        categories=categories,
        payees=payees,
        metadata={
            "by_category": by_category,
            "top_payees": top_payees,
            "big_expenses": big_expenses,
            "income_payee_distribution": income_payee_distribution,
        },
    )


def _extract_weekly_report_document(path: Path) -> FinanceDocument:
    content = path.read_text(encoding="utf-8")
    report_date = path.stem.replace("weekly_report_", "")
    title = content.splitlines()[0].lstrip("# ").strip() if content.strip() else f"Weekly report {report_date}"
    return FinanceDocument(
        doc_id=f"weekly_report::{path.stem}",
        doc_type="weekly_report",
        source_path=str(path),
        title=title,
        content=content,
        start_date=None,
        end_date=report_date,
        total_income=None,
        total_expense=None,
        net_cashflow=None,
        categories=[],
        payees=[],
        metadata={"report_date": report_date},
    )


def _iter_artifact_paths(base_dir: Path) -> Iterable[tuple[str, Path]]:
    patterns = {
        "daily_snapshot": base_dir / "daily_snapshots",
        "weekly_snapshot": base_dir / "weekly_snapshots",
        "weekly_report": base_dir / "weekly_reports",
    }
    for doc_type, folder in patterns.items():
        if not folder.exists():
            continue
        suffix = "*.md" if doc_type == "weekly_report" else "*.json"
        for path in sorted(folder.glob(suffix)):
            yield doc_type, path


def _document_from_path(doc_type: str, path: Path) -> FinanceDocument:
    if doc_type == "daily_snapshot":
        return _extract_daily_document(path)
    if doc_type == "weekly_snapshot":
        return _extract_weekly_snapshot_document(path)
    if doc_type == "weekly_report":
        return _extract_weekly_report_document(path)
    raise ValueError(f"Unsupported document type: {doc_type}")


def rebuild_document_store(base_dir: str = ".", db_path: Optional[str] = None) -> Dict[str, int]:
    init_document_store(db_path)
    docs = [_document_from_path(doc_type, path) for doc_type, path in _iter_artifact_paths(Path(base_dir))]
    with get_document_connection(db_path) as conn:
        conn.execute("DELETE FROM finance_documents")
        conn.executemany(
            """
            INSERT INTO finance_documents (
                doc_id, doc_type, source_path, title, content, start_date, end_date,
                total_income, total_expense, net_cashflow, categories_json, payees_json,
                metadata_json, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    doc.doc_id,
                    doc.doc_type,
                    doc.source_path,
                    doc.title,
                    doc.content,
                    doc.start_date,
                    doc.end_date,
                    doc.total_income,
                    doc.total_expense,
                    doc.net_cashflow,
                    _json_text(doc.categories),
                    _json_text(doc.payees),
                    _json_text(doc.metadata),
                    _content_hash(doc.content),
                )
                for doc in docs
            ],
        )
    counts: Dict[str, int] = {}
    for doc in docs:
        counts[doc.doc_type] = counts.get(doc.doc_type, 0) + 1
    counts["total"] = len(docs)
    return counts


def search_documents(
    query: Optional[str] = None,
    doc_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    init_document_store(db_path)
    sql = """
        SELECT
            doc_id, doc_type, source_path, title, content, start_date, end_date,
            total_income, total_expense, net_cashflow, categories_json, payees_json, metadata_json
        FROM finance_documents
        WHERE 1=1
    """
    params: List[Any] = []
    if doc_type:
        sql += " AND doc_type = ?"
        params.append(doc_type)
    if start_date:
        sql += " AND COALESCE(end_date, start_date) >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND COALESCE(start_date, end_date) <= ?"
        params.append(end_date)
    if query:
        sql += " AND (title LIKE ? OR content LIKE ? OR categories_json LIKE ? OR payees_json LIKE ?)"
        needle = f"%{query}%"
        params.extend([needle, needle, needle, needle])
    sql += " ORDER BY COALESCE(end_date, start_date) DESC, title ASC LIMIT ?"
    params.append(limit)

    with get_document_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "doc_id": row["doc_id"],
                "doc_type": row["doc_type"],
                "source_path": row["source_path"],
                "title": row["title"],
                "content": row["content"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "total_income": row["total_income"],
                "total_expense": row["total_expense"],
                "net_cashflow": row["net_cashflow"],
                "categories": json.loads(row["categories_json"]),
                "payees": json.loads(row["payees_json"]),
                "metadata": json.loads(row["metadata_json"]),
            }
        )
    return results


def _all_documents(
    doc_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return search_documents(
        query=None,
        doc_type=doc_type,
        start_date=start_date,
        end_date=end_date,
        limit=10_000,
        db_path=db_path,
    )


def _normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _matches_text(candidates: Iterable[str], needle: str) -> bool:
    normalized_needle = _normalize_text(needle)
    return any(_normalize_text(candidate) == normalized_needle for candidate in candidates)


def search_past_weeks_by_category(
    category: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    matches = []
    for doc in _all_documents(
        doc_type="weekly_snapshot",
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
    ):
        if not _matches_text(doc["categories"], category):
            continue
        amount = None
        for item in doc["metadata"].get("by_category", []):
            if _normalize_text(item.get("category")) == _normalize_text(category):
                amount = item.get("amount")
                break
        matches.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "start_date": doc["start_date"],
                "end_date": doc["end_date"],
                "category": category,
                "category_amount": amount,
                "total_expense": doc["total_expense"],
                "net_cashflow": doc["net_cashflow"],
                "source_path": doc["source_path"],
            }
        )
    return matches[:limit]


def find_similar_spending_weeks(
    start_date: str,
    end_date: str,
    limit: int = 3,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    weekly_docs = _all_documents(doc_type="weekly_snapshot", db_path=db_path)
    target = next(
        (
            doc
            for doc in weekly_docs
            if doc["start_date"] == start_date and doc["end_date"] == end_date
        ),
        None,
    )
    if target is None:
        return []

    target_categories = set(target["categories"])
    target_expense = float(target["total_expense"] or 0.0)
    target_income = float(target["total_income"] or 0.0)
    target_net = float(target["net_cashflow"] or 0.0)

    scored = []
    for doc in weekly_docs:
        if doc["doc_id"] == target["doc_id"]:
            continue
        doc_categories = set(doc["categories"])
        overlap = len(target_categories & doc_categories)
        expense_gap = abs(float(doc["total_expense"] or 0.0) - target_expense)
        income_gap = abs(float(doc["total_income"] or 0.0) - target_income)
        net_gap = abs(float(doc["net_cashflow"] or 0.0) - target_net)
        score = (overlap * 1000) - expense_gap - (income_gap * 0.5) - (net_gap * 0.25)
        scored.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "start_date": doc["start_date"],
                "end_date": doc["end_date"],
                "total_income": doc["total_income"],
                "total_expense": doc["total_expense"],
                "net_cashflow": doc["net_cashflow"],
                "shared_categories": sorted(target_categories & doc_categories),
                "similarity_score": round(score, 2),
                "source_path": doc["source_path"],
            }
        )
    scored.sort(key=lambda item: item["similarity_score"], reverse=True)
    return scored[:limit]


def get_recent_anomalies(
    payee: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    anomalies = []
    for doc in _all_documents(doc_type="weekly_snapshot", db_path=db_path):
        for item in doc["metadata"].get("big_expenses", []):
            if payee and _normalize_text(item.get("payee")) != _normalize_text(payee):
                continue
            if category and _normalize_text(item.get("category")) != _normalize_text(category):
                continue
            anomalies.append(
                {
                    "date": item.get("date"),
                    "payee": item.get("payee"),
                    "category": item.get("category"),
                    "amount": item.get("amount"),
                    "week_start": doc["start_date"],
                    "week_end": doc["end_date"],
                    "source_path": doc["source_path"],
                }
            )
    anomalies.sort(key=lambda item: (item.get("date") or "", item.get("amount") or 0), reverse=True)
    return anomalies[:limit]


def search_reports(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    results = search_documents(
        query=query,
        doc_type="weekly_report",
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        db_path=db_path,
    )
    return [
        {
            "doc_id": item["doc_id"],
            "title": item["title"],
            "end_date": item["end_date"],
            "excerpt": item["content"][:500],
            "source_path": item["source_path"],
        }
        for item in results
    ]
