from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_CONVERSATION_DB_PATH = "db.sqlite"


def get_conversation_db_path(db_path: Optional[str] = None) -> str:
    return db_path or os.getenv("ACTUAL_DB_PATH", DEFAULT_CONVERSATION_DB_PATH)


def get_conversation_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(get_conversation_db_path(db_path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_conversations (
            conversation_id TEXT PRIMARY KEY,
            account_pid TEXT,
            account_name TEXT,
            card_label TEXT,
            context_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_message_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES chat_conversations(conversation_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated ON chat_conversations(updated_at)")
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _context_from_request(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return context or {}


def _upsert_conversation_in_connection(
    conn: sqlite3.Connection,
    conversation_id: str,
    context_dict: Dict[str, Any],
) -> None:
    now = _now_iso()
    existing = conn.execute(
        "SELECT created_at FROM chat_conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE chat_conversations
            SET account_pid = ?, account_name = ?, card_label = ?, context_json = ?, updated_at = ?, last_message_at = ?
            WHERE conversation_id = ?
            """,
            (
                context_dict.get("account_pid"),
                context_dict.get("account_name"),
                context_dict.get("card_label"),
                json.dumps(context_dict, ensure_ascii=False),
                now,
                now,
                conversation_id,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO chat_conversations (
                conversation_id, account_pid, account_name, card_label, context_json, created_at, updated_at, last_message_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                context_dict.get("account_pid"),
                context_dict.get("account_name"),
                context_dict.get("card_label"),
                json.dumps(context_dict, ensure_ascii=False),
                now,
                now,
                now,
            ),
        )


def upsert_conversation(
    conversation_id: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> None:
    context_dict = _context_from_request(context)
    with get_conversation_connection(db_path) as conn:
        _upsert_conversation_in_connection(conn, conversation_id, context_dict)


def append_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> None:
    now = _now_iso()
    with get_conversation_connection(db_path) as conn:
        _upsert_conversation_in_connection(conn, conversation_id, _context_from_request(context))
        conn.execute(
            """
            INSERT INTO chat_messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, now),
        )
        conn.execute(
            """
            UPDATE chat_conversations
            SET updated_at = ?, last_message_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )


def list_messages(conversation_id: str, *, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_conversation_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [
        {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
        for row in rows
    ]


def load_conversation(conversation_id: str, *, db_path: Optional[str] = None) -> Dict[str, Any]:
    with get_conversation_connection(db_path) as conn:
        conversation = conn.execute(
            """
            SELECT conversation_id, account_pid, account_name, card_label, context_json, created_at, updated_at, last_message_at
            FROM chat_conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise KeyError(conversation_id)
        messages = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    context_json = conversation["context_json"] or "{}"
    try:
        context = json.loads(context_json)
    except json.JSONDecodeError:
        context = {}

    return {
        "conversation_id": conversation["conversation_id"],
        "account_pid": conversation["account_pid"],
        "account_name": conversation["account_name"],
        "card_label": conversation["card_label"],
        "context": context,
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "last_message_at": conversation["last_message_at"],
        "messages": [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in messages
        ],
    }


def list_conversations(
    account_pid: Optional[str] = None,
    *,
    limit: int = 10,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            c.conversation_id,
            c.account_pid,
            c.account_name,
            c.card_label,
            c.updated_at,
            c.last_message_at,
            c.context_json,
            COUNT(m.id) AS message_count,
            (
                SELECT cm.content
                FROM chat_messages cm
                WHERE cm.conversation_id = c.conversation_id
                ORDER BY cm.id DESC
                LIMIT 1
            ) AS last_message
        FROM chat_conversations c
        LEFT JOIN chat_messages m ON m.conversation_id = c.conversation_id
    """
    params: list[Any] = []
    if account_pid:
        query += " WHERE c.account_pid = ?"
        params.append(account_pid)
    query += """
        GROUP BY c.conversation_id, c.account_pid, c.account_name, c.card_label, c.updated_at, c.last_message_at, c.context_json
        ORDER BY c.last_message_at DESC
        LIMIT ?
    """
    params.append(limit)

    with get_conversation_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    conversations: List[Dict[str, Any]] = []
    for row in rows:
        context_json = row["context_json"] or "{}"
        try:
            context = json.loads(context_json)
        except json.JSONDecodeError:
            context = {}
        conversations.append(
            {
                "conversation_id": row["conversation_id"],
                "account_pid": row["account_pid"],
                "account_name": row["account_name"],
                "card_label": row["card_label"] or context.get("card_label") or context.get("card") or row["account_name"],
                "updated_at": row["updated_at"],
                "last_message_at": row["last_message_at"],
                "message_count": int(row["message_count"] or 0),
                "preview": (row["last_message"] or "").strip(),
            }
        )
    return conversations
