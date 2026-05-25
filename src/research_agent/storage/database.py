"""SQLite database and PaperRepository.

Schema follows ``planning/architecture.md §4.2`` with two pragmatic extensions
(``sections_json`` and ``full_text``) so that parsed papers can be restored
without re-running the PDF parser.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

from research_agent.core.paper import Paper, Section

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    year            INTEGER,
    venue           TEXT,
    pdf_path        TEXT,
    sections_json   TEXT,
    full_text       TEXT,
    analyst_notes   TEXT,
    critic_notes    TEXT,
    relevance_score REAL,
    read_at         TIMESTAMP,
    tags            TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);

CREATE TABLE IF NOT EXISTS ideas (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT,
    status       TEXT,
    analyst_score    REAL,
    critic_score     REAL,
    critic_objections TEXT,
    related_papers   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS discussions (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_discussions_session ON discussions(session_id);
"""


class Database:
    """Thin wrapper around a ``sqlite3`` connection with auto-initialised schema."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class PaperRepository:
    """CRUD operations for :class:`Paper`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, paper: Paper) -> None:
        """Insert-or-replace a paper by id."""
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO papers (
                    id, title, authors, abstract, year, venue, pdf_path,
                    sections_json, full_text, tags
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    authors = excluded.authors,
                    abstract = excluded.abstract,
                    year = excluded.year,
                    venue = excluded.venue,
                    pdf_path = excluded.pdf_path,
                    sections_json = excluded.sections_json,
                    full_text = excluded.full_text,
                    tags = excluded.tags
                """,
                (
                    paper.id,
                    paper.title,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.abstract,
                    paper.year,
                    paper.venue,
                    str(paper.pdf_path) if paper.pdf_path else None,
                    json.dumps(
                        [{"title": s.title, "content": s.content} for s in paper.sections],
                        ensure_ascii=False,
                    ),
                    paper.full_text,
                    json.dumps(paper.tags, ensure_ascii=False),
                ),
            )

    def get(self, paper_id: str) -> Paper | None:
        row = self.db.conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return _row_to_paper(row) if row else None

    def find_by_title(self, query: str) -> list[Paper]:
        rows = self.db.conn.execute(
            "SELECT * FROM papers WHERE title LIKE ? ORDER BY title",
            (f"%{query}%",),
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    def find_by_tag(self, tag: str) -> list[Paper]:
        like = f'%"{tag}"%'
        rows = self.db.conn.execute(
            "SELECT * FROM papers WHERE tags LIKE ? ORDER BY title",
            (like,),
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    def list_all(self) -> list[Paper]:
        rows = self.db.conn.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
        return [_row_to_paper(r) for r in rows]

    def delete(self, paper_id: str) -> bool:
        with self.db.conn:
            cur = self.db.conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        return cur.rowcount > 0


def _row_to_paper(row: sqlite3.Row) -> Paper:
    sections_raw = row["sections_json"]
    sections: list[Section] = []
    if sections_raw:
        sections = [Section(**s) for s in json.loads(sections_raw)]
    return Paper(
        id=row["id"],
        title=row["title"],
        authors=json.loads(row["authors"]) if row["authors"] else [],
        abstract=row["abstract"] or "",
        sections=sections,
        full_text=row["full_text"] or "",
        year=row["year"],
        venue=row["venue"],
        pdf_path=Path(row["pdf_path"]) if row["pdf_path"] else None,
        tags=json.loads(row["tags"]) if row["tags"] else [],
    )
