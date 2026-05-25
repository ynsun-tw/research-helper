"""Discussion session persistence (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from research_agent.storage.database import Database


@dataclass(slots=True)
class DiscussionMessage:
    id: str
    session_id: str
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DiscussionRepository:
    """Append-only store for interactive discuss sessions."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> str:
        mid = message_id or str(uuid.uuid4())
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO discussions (id, session_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (mid, session_id, role, content, meta_json),
            )
        return mid

    def list_session(self, session_id: str) -> list[DiscussionMessage]:
        rows = self.db.conn.execute(
            "SELECT * FROM discussions WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [_row_to_message(r) for r in rows]


def _row_to_message(row: sqlite3.Row) -> DiscussionMessage:
    meta_raw = row["metadata"]
    metadata: dict[str, Any] = json.loads(meta_raw) if meta_raw else {}
    return DiscussionMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        metadata=metadata,
    )
