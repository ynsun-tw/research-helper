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
    activation_conditions TEXT,
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

-- M3.1 (precursor to the full Searcher Agent): keep a lightweight log of
-- queries the user has issued and the hits returned so /search has memory
-- across REPL sessions and we can flag already-read papers.
CREATE TABLE IF NOT EXISTS search_queries (
    id          TEXT PRIMARY KEY,
    session_id  TEXT,
    query       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'arxiv',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_search_queries_created
    ON search_queries(created_at);

CREATE TABLE IF NOT EXISTS search_results (
    id                TEXT PRIMARY KEY,
    query_id          TEXT NOT NULL,
    arxiv_id          TEXT NOT NULL,
    title             TEXT NOT NULL,
    abstract          TEXT,
    published         TEXT,
    rank              INTEGER,
    relevance_score   REAL,
    relevance_reason  TEXT,
    FOREIGN KEY (query_id) REFERENCES search_queries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_search_results_arxiv
    ON search_results(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_search_results_query
    ON search_results(query_id);

CREATE TABLE IF NOT EXISTS reading_queue (
    id            TEXT PRIMARY KEY,
    arxiv_id      TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'manual',
    status        TEXT NOT NULL DEFAULT 'pending',
    notes         TEXT,
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at  TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reading_queue_status
    ON reading_queue(status, added_at);

-- M4 S4.1.1: paragraph-level prose samples extracted from the user's own
-- papers (or any papers they want Scribe to mimic). Each row is one
-- filtered paragraph; multiple rows per source paper. Foreign-key to
-- ``papers.id`` is intentionally not enforced - style training works
-- equally well on papers the user never /read'd inside the REPL.
CREATE TABLE IF NOT EXISTS style_samples (
    id            TEXT PRIMARY KEY,
    paper_id      TEXT NOT NULL,
    section_title TEXT NOT NULL DEFAULT '',
    paragraph     TEXT NOT NULL,
    char_count    INTEGER NOT NULL DEFAULT 0,
    word_count    INTEGER NOT NULL DEFAULT 0,
    sentence_count INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_style_samples_paper
    ON style_samples(paper_id);

-- M4 S4.3.2: (original, revised) draft pairs harvested from the
-- interactive writing-review loop. The ``selected_*`` / ``rejected_*``
-- columns let S4.1.3 know which reviewer feedback the user chose to
-- act on - the diff that follows is the signal we want to learn from.
CREATE TABLE IF NOT EXISTS draft_revisions (
    id                    TEXT PRIMARY KEY,
    section               TEXT NOT NULL,
    original_text         TEXT NOT NULL,
    revised_text          TEXT NOT NULL,
    selected_issues       TEXT NOT NULL DEFAULT '[]',
    selected_suggestions  TEXT NOT NULL DEFAULT '[]',
    rejected_issues       TEXT NOT NULL DEFAULT '[]',
    rejected_suggestions  TEXT NOT NULL DEFAULT '[]',
    interactive           INTEGER NOT NULL DEFAULT 0,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_draft_revisions_created
    ON draft_revisions(created_at);
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
            self._migrate()

    def _migrate(self) -> None:
        """Add M2 columns to existing databases without breaking older installs."""
        idea_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(ideas)").fetchall()}
        if "score_history" not in idea_cols:
            self.conn.execute("ALTER TABLE ideas ADD COLUMN score_history TEXT")
        if "user_score_feedback" not in idea_cols:
            self.conn.execute("ALTER TABLE ideas ADD COLUMN user_score_feedback TEXT")
        if "activation_conditions" not in idea_cols:
            self.conn.execute("ALTER TABLE ideas ADD COLUMN activation_conditions TEXT")

        disc_rows = self.conn.execute("PRAGMA table_info(discussions)").fetchall()
        disc_cols = {row[1] for row in disc_rows}
        if "idea_id" not in disc_cols:
            self.conn.execute("ALTER TABLE discussions ADD COLUMN idea_id TEXT")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_discussions_idea ON discussions(idea_id)"
            )

        # M3 task 1: relevance scoring on search_results
        sr_rows = self.conn.execute("PRAGMA table_info(search_results)").fetchall()
        sr_cols = {row[1] for row in sr_rows}
        if sr_rows and "relevance_score" not in sr_cols:
            self.conn.execute("ALTER TABLE search_results ADD COLUMN relevance_score REAL")
        if sr_rows and "relevance_reason" not in sr_cols:
            self.conn.execute("ALTER TABLE search_results ADD COLUMN relevance_reason TEXT")

        # M5 S5.4.1: query-pattern indexes. We keep these out of the
        # declarative SCHEMA so legacy databases (which may pre-date
        # the ``created_at`` / ``updated_at`` columns referenced
        # below) don't crash at open time. Each index is gated on the
        # target columns actually existing.
        idea_cols_now = {
            row[1] for row in self.conn.execute("PRAGMA table_info(ideas)").fetchall()
        }
        if {"updated_at", "created_at"}.issubset(idea_cols_now):
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ideas_updated "
                "ON ideas(updated_at DESC, created_at DESC)"
            )
        if "status" in idea_cols_now:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status)"
            )
        if "created_at" in idea_cols_now:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ideas_created ON ideas(created_at)"
            )

        paper_cols = {
            row[1] for row in self.conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        if "created_at" in paper_cols:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_created "
                "ON papers(created_at DESC)"
            )

        disc_cols_now = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(discussions)").fetchall()
        }
        if {"session_id", "created_at"}.issubset(disc_cols_now):
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_discussions_session_created "
                "ON discussions(session_id, created_at)"
            )
        if "created_at" in disc_cols_now:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_discussions_created "
                "ON discussions(created_at)"
            )

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

    def save_analysis_notes(
        self,
        paper_id: str,
        *,
        analyst_notes: dict[str, object] | None = None,
        critic_notes: dict[str, object] | None = None,
    ) -> None:
        """Persist serialized Analyst/Critic results on the paper row."""
        with self.db.conn:
            if analyst_notes is not None:
                self.db.conn.execute(
                    "UPDATE papers SET analyst_notes = ? WHERE id = ?",
                    (json.dumps(analyst_notes, ensure_ascii=False), paper_id),
                )
            if critic_notes is not None:
                self.db.conn.execute(
                    "UPDATE papers SET critic_notes = ? WHERE id = ?",
                    (json.dumps(critic_notes, ensure_ascii=False), paper_id),
                )


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
