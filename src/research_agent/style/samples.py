"""``style_samples`` table model + repository.

Each row stores ONE filtered prose paragraph from ONE source paper. We
keep the section title and a few cheap counts (chars / words /
sentences) on the row so the macro/micro fingerprint analyzer in
S4.1.2 can roll them up without re-parsing the paragraph.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from research_agent.storage.database import Database


@dataclass(slots=True)
class StyleSample:
    """One paragraph of training text + its metadata."""

    id: str
    paper_id: str
    section_title: str
    paragraph: str
    char_count: int
    word_count: int
    sentence_count: int
    created_at: str = ""


class StyleSampleRepository:
    """SQLite CRUD for :class:`StyleSample`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        *,
        paper_id: str,
        section_title: str,
        paragraph: str,
        char_count: int,
        word_count: int,
        sentence_count: int,
        sample_id: str | None = None,
    ) -> StyleSample:
        sid = sample_id or str(uuid.uuid4())
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO style_samples (
                    id, paper_id, section_title, paragraph,
                    char_count, word_count, sentence_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    paper_id,
                    section_title,
                    paragraph,
                    char_count,
                    word_count,
                    sentence_count,
                ),
            )
        return StyleSample(
            id=sid,
            paper_id=paper_id,
            section_title=section_title,
            paragraph=paragraph,
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
        )

    def bulk_add(self, samples: list[StyleSample]) -> int:
        """Insert many samples in one transaction. Returns the row count."""
        if not samples:
            return 0
        rows = [
            (
                s.id or str(uuid.uuid4()),
                s.paper_id,
                s.section_title,
                s.paragraph,
                s.char_count,
                s.word_count,
                s.sentence_count,
            )
            for s in samples
        ]
        with self.db.conn:
            self.db.conn.executemany(
                """
                INSERT INTO style_samples (
                    id, paper_id, section_title, paragraph,
                    char_count, word_count, sentence_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def list_all(self) -> list[StyleSample]:
        rows = self.db.conn.execute(
            "SELECT * FROM style_samples ORDER BY created_at, id"
        ).fetchall()
        return [_row_to_sample(r) for r in rows]

    def iter_all(self) -> Iterator[StyleSample]:
        """Stream every sample - cheaper for large corpora."""
        for row in self.db.conn.execute(
            "SELECT * FROM style_samples ORDER BY created_at, id"
        ):
            yield _row_to_sample(row)

    def for_paper(self, paper_id: str) -> list[StyleSample]:
        rows = self.db.conn.execute(
            "SELECT * FROM style_samples WHERE paper_id = ? ORDER BY created_at",
            (paper_id,),
        ).fetchall()
        return [_row_to_sample(r) for r in rows]

    def count(self) -> int:
        cur = self.db.conn.execute("SELECT COUNT(*) FROM style_samples")
        return int(cur.fetchone()[0])

    def count_by_paper(self) -> dict[str, int]:
        """Return {paper_id: paragraph_count}."""
        rows = self.db.conn.execute(
            "SELECT paper_id, COUNT(*) FROM style_samples GROUP BY paper_id"
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def delete_for_paper(self, paper_id: str) -> int:
        """Remove all samples from one source. Returns deleted row count.

        Used by ``style train`` when re-importing a paper: we want
        the new run to replace the old samples, not duplicate them.
        """
        with self.db.conn:
            cur = self.db.conn.execute(
                "DELETE FROM style_samples WHERE paper_id = ?",
                (paper_id,),
            )
        return cur.rowcount


def _row_to_sample(row) -> StyleSample:  # type: ignore[no-untyped-def]
    return StyleSample(
        id=row["id"],
        paper_id=row["paper_id"],
        section_title=row["section_title"] or "",
        paragraph=row["paragraph"],
        char_count=int(row["char_count"] or 0),
        word_count=int(row["word_count"] or 0),
        sentence_count=int(row["sentence_count"] or 0),
        created_at=str(row["created_at"] or ""),
    )
