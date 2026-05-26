"""Reading queue persistence (M3 task 3).

A lightweight "to-read" list for arXiv papers the user has saved but not
analysed yet. Entries flow through ``pending → in_progress → done`` (or
``skipped``) and back the ``/read --queue`` and ``queue_*`` LLM tools.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from typing import Literal

from research_agent.storage.database import Database

QueueStatus = Literal["pending", "in_progress", "done", "skipped"]
ALLOWED_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "done", "skipped"}
)


@dataclass(slots=True)
class QueueEntry:
    id: str
    arxiv_id: str
    title: str
    source: str
    status: QueueStatus
    notes: str
    added_at: str
    completed_at: str | None


class ReadingQueueRepository:
    """CRUD over the ``reading_queue`` SQLite table."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------------------------- mutators ---------------------------------

    def add(
        self,
        arxiv_id: str,
        *,
        title: str = "",
        source: str = "manual",
        notes: str = "",
    ) -> QueueEntry:
        """Insert or refresh a pending entry. Returns the row, new or existing.

        If an entry exists (any status), we leave its state alone but refresh
        the title/source/notes so re-adding from a fresh /search result fills
        in metadata that was missing on the original manual add.
        """
        arxiv_id = arxiv_id.strip()
        if not arxiv_id:
            raise ValueError("arxiv_id is required")
        existing = self.get(arxiv_id)
        if existing is not None:
            updates: list[str] = []
            params: list[str] = []
            if title and title != existing.title:
                updates.append("title = ?")
                params.append(title)
            if source and source != existing.source:
                updates.append("source = ?")
                params.append(source)
            if notes and notes != existing.notes:
                updates.append("notes = ?")
                params.append(notes)
            if updates:
                params.append(existing.id)
                with self.db.conn:
                    self.db.conn.execute(
                        f"UPDATE reading_queue SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
            return self.get(arxiv_id) or existing

        row_id = str(uuid.uuid4())
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO reading_queue (id, arxiv_id, title, source, status, notes)
                VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (row_id, arxiv_id, title, source, notes),
            )
        entry = self.get(arxiv_id)
        assert entry is not None
        return entry

    def remove(self, arxiv_id: str) -> bool:
        with self.db.conn:
            cur = self.db.conn.execute(
                "DELETE FROM reading_queue WHERE arxiv_id = ?", (arxiv_id,)
            )
        return cur.rowcount > 0

    def set_status(
        self, arxiv_id: str, status: QueueStatus, *, notes: str | None = None
    ) -> QueueEntry | None:
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                f"invalid status {status!r}; "
                f"expected one of {sorted(ALLOWED_STATUSES)}"
            )
        if self.get(arxiv_id) is None:
            return None
        completed_at_sql = (
            "CURRENT_TIMESTAMP" if status in {"done", "skipped"} else "NULL"
        )
        if notes is None:
            with self.db.conn:
                self.db.conn.execute(
                    f"""
                    UPDATE reading_queue
                       SET status = ?, completed_at = {completed_at_sql}
                     WHERE arxiv_id = ?
                    """,
                    (status, arxiv_id),
                )
        else:
            with self.db.conn:
                self.db.conn.execute(
                    f"""
                    UPDATE reading_queue
                       SET status = ?, notes = ?, completed_at = {completed_at_sql}
                     WHERE arxiv_id = ?
                    """,
                    (status, notes, arxiv_id),
                )
        return self.get(arxiv_id)

    # ----------------------------- queries ---------------------------------

    def get(self, arxiv_id: str) -> QueueEntry | None:
        row = self.db.conn.execute(
            "SELECT * FROM reading_queue WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def list(self, status: QueueStatus | None = None) -> list[QueueEntry]:
        if status is None:
            rows = self.db.conn.execute(
                "SELECT * FROM reading_queue ORDER BY added_at ASC"
            ).fetchall()
        else:
            if status not in ALLOWED_STATUSES:
                raise ValueError(f"invalid status {status!r}")
            rows = self.db.conn.execute(
                "SELECT * FROM reading_queue WHERE status = ? ORDER BY added_at ASC",
                (status,),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def next_pending(self) -> QueueEntry | None:
        row = self.db.conn.execute(
            """
            SELECT * FROM reading_queue
             WHERE status = 'pending'
             ORDER BY added_at ASC
             LIMIT 1
            """
        ).fetchone()
        return _row_to_entry(row) if row else None


def _row_to_entry(row: sqlite3.Row) -> QueueEntry:
    return QueueEntry(
        id=row["id"],
        arxiv_id=row["arxiv_id"],
        title=row["title"] or "",
        source=row["source"] or "manual",
        status=row["status"],
        notes=row["notes"] or "",
        added_at=row["added_at"],
        completed_at=row["completed_at"],
    )
