"""SQLite repository for (original, revised) draft pairs (M4 S4.3.2).

Every time the user accepts (or partially accepts) a Scribe revision
we persist the before/after plus which reviewer suggestions they
chose to act on. S4.1.3 (continuous fingerprint learning) consumes
this corpus to refresh the style fingerprint over time without
needing a fresh ``research style train`` run.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from research_agent.storage.database import Database


@dataclass
class DraftRevision:
    """One (original, revised) pair with the user's selection trace."""

    id: str
    section: str
    original_text: str
    revised_text: str
    selected_issues: list[str] = field(default_factory=list)
    selected_suggestions: list[str] = field(default_factory=list)
    rejected_issues: list[str] = field(default_factory=list)
    rejected_suggestions: list[str] = field(default_factory=list)
    interactive: bool = False
    created_at: str = ""


class DraftRevisionRepository:
    """CRUD on the ``draft_revisions`` table."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        *,
        section: str,
        original_text: str,
        revised_text: str,
        selected_issues: list[str] | None = None,
        selected_suggestions: list[str] | None = None,
        rejected_issues: list[str] | None = None,
        rejected_suggestions: list[str] | None = None,
        interactive: bool = False,
        revision_id: str | None = None,
    ) -> DraftRevision:
        rid = revision_id or str(uuid.uuid4())
        sel_i = selected_issues or []
        sel_s = selected_suggestions or []
        rej_i = rejected_issues or []
        rej_s = rejected_suggestions or []
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO draft_revisions (
                    id, section, original_text, revised_text,
                    selected_issues, selected_suggestions,
                    rejected_issues, rejected_suggestions,
                    interactive
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    section,
                    original_text,
                    revised_text,
                    json.dumps(sel_i, ensure_ascii=False),
                    json.dumps(sel_s, ensure_ascii=False),
                    json.dumps(rej_i, ensure_ascii=False),
                    json.dumps(rej_s, ensure_ascii=False),
                    1 if interactive else 0,
                ),
            )
        return DraftRevision(
            id=rid,
            section=section,
            original_text=original_text,
            revised_text=revised_text,
            selected_issues=sel_i,
            selected_suggestions=sel_s,
            rejected_issues=rej_i,
            rejected_suggestions=rej_s,
            interactive=interactive,
        )

    def get(self, revision_id: str) -> DraftRevision | None:
        row = self.db.conn.execute(
            "SELECT * FROM draft_revisions WHERE id = ?", (revision_id,)
        ).fetchone()
        return _row_to_revision(row) if row else None

    def list_all(self) -> list[DraftRevision]:
        rows = self.db.conn.execute(
            "SELECT * FROM draft_revisions ORDER BY created_at"
        ).fetchall()
        return [_row_to_revision(r) for r in rows]

    def iter_all(self) -> Iterator[DraftRevision]:
        for row in self.db.conn.execute(
            "SELECT * FROM draft_revisions ORDER BY created_at"
        ):
            yield _row_to_revision(row)

    def count(self) -> int:
        cur = self.db.conn.execute("SELECT COUNT(*) FROM draft_revisions")
        return int(cur.fetchone()[0])


def _row_to_revision(row) -> DraftRevision:  # type: ignore[no-untyped-def]
    return DraftRevision(
        id=row["id"],
        section=row["section"] or "",
        original_text=row["original_text"] or "",
        revised_text=row["revised_text"] or "",
        selected_issues=_load_list(row["selected_issues"]),
        selected_suggestions=_load_list(row["selected_suggestions"]),
        rejected_issues=_load_list(row["rejected_issues"]),
        rejected_suggestions=_load_list(row["rejected_suggestions"]),
        interactive=bool(row["interactive"]),
        created_at=str(row["created_at"] or ""),
    )


def _load_list(value: object) -> list[str]:
    """Tolerant JSON-list parser. Treats missing / malformed as ``[]``."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        data = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(v) for v in data]
