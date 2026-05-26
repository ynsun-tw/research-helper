"""Persistence layer for /search history (M3.1 precursor to Searcher Agent).

Stores one ``search_queries`` row per /search call plus one ``search_results``
row per hit returned. Used by the chat REPL to:

- show ``/history`` across sessions
- mark hits that the user has already loaded via /read (``read=True``)

The full Searcher Agent (M3 E3.1/E3.3) layers relevance scoring + multi-source
search on top of this table; we keep the schema narrow on purpose so future
sources can extend with a ``source`` column rather than a new table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from research_agent.search.arxiv_search import ArxivSearchHit
from research_agent.storage.database import Database


@dataclass(slots=True)
class StoredSearchHit:
    arxiv_id: str
    title: str
    abstract: str
    published: str
    rank: int
    relevance_score: float | None = None
    relevance_reason: str = ""
    read: bool = False  # populated by SearchRepository.recent_hits via join

    def to_arxiv_hit(self) -> ArxivSearchHit:
        return ArxivSearchHit(
            arxiv_id=self.arxiv_id,
            title=self.title,
            abstract=self.abstract,
            published=self.published,
            relevance_score=self.relevance_score,
            relevance_reason=self.relevance_reason,
        )


@dataclass(slots=True)
class StoredSearchQuery:
    id: str
    query: str
    source: str
    created_at: str
    session_id: str | None
    hits: list[StoredSearchHit]


class SearchRepository:
    """CRUD for ``search_queries`` + ``search_results`` tables."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        query: str,
        hits: list[ArxivSearchHit],
        *,
        source: str = "arxiv",
        session_id: str | None = None,
    ) -> str:
        """Insert a new query + its hits in one transaction. Returns query id.

        Hits may carry ``relevance_score`` / ``relevance_reason`` (set by the
        Searcher agent); when present, ``rank`` is assigned by descending score
        so /history and recent_searches surface the strongest matches first.
        """
        query_id = str(uuid.uuid4())
        ordered = list(enumerate(hits))
        if any(h.relevance_score is not None for h in hits):
            ordered.sort(
                key=lambda pair: (
                    -(pair[1].relevance_score or 0.0),
                    pair[0],
                )
            )
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO search_queries (id, session_id, query, source)
                VALUES (?, ?, ?, ?)
                """,
                (query_id, session_id, query, source),
            )
            for rank, (_, hit) in enumerate(ordered, start=1):
                self.db.conn.execute(
                    """
                    INSERT INTO search_results
                        (id, query_id, arxiv_id, title, abstract, published,
                         rank, relevance_score, relevance_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        query_id,
                        hit.arxiv_id,
                        hit.title,
                        hit.abstract,
                        hit.published,
                        rank,
                        hit.relevance_score,
                        hit.relevance_reason,
                    ),
                )
        return query_id

    def already_read(self, arxiv_ids: list[str]) -> set[str]:
        """Return the subset of given arXiv ids that have a ``papers`` row.

        Matches both ``arxiv:<id>`` and bare ``<id>`` storage formats.
        """
        if not arxiv_ids:
            return set()
        candidates: list[str] = []
        for a in arxiv_ids:
            candidates.append(a)
            candidates.append(f"arxiv:{a}")
        placeholders = ",".join("?" for _ in candidates)
        rows = self.db.conn.execute(
            f"SELECT id FROM papers WHERE id IN ({placeholders})",
            candidates,
        ).fetchall()
        out: set[str] = set()
        for row in rows:
            paper_id = str(row["id"])
            out.add(paper_id.removeprefix("arxiv:"))
        return out

    def recent_queries(self, limit: int = 20) -> list[StoredSearchQuery]:
        """Return the N most recent search queries, newest first, with hits attached."""
        rows = self.db.conn.execute(
            """
            SELECT id, session_id, query, source, created_at
              FROM search_queries
             ORDER BY datetime(created_at) DESC, id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        if not rows:
            return []

        query_ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in query_ids)
        hit_rows = self.db.conn.execute(
            f"""
            SELECT query_id, arxiv_id, title, abstract, published, rank,
                   relevance_score, relevance_reason
              FROM search_results
             WHERE query_id IN ({placeholders})
             ORDER BY rank ASC
            """,
            query_ids,
        ).fetchall()
        already_read = self.already_read(
            list({h["arxiv_id"] for h in hit_rows})
        )

        by_query: dict[str, list[StoredSearchHit]] = {qid: [] for qid in query_ids}
        for h in hit_rows:
            relevance = h["relevance_score"]
            by_query[h["query_id"]].append(
                StoredSearchHit(
                    arxiv_id=h["arxiv_id"],
                    title=h["title"],
                    abstract=h["abstract"] or "",
                    published=h["published"] or "",
                    rank=int(h["rank"]) if h["rank"] is not None else 0,
                    relevance_score=float(relevance) if relevance is not None else None,
                    relevance_reason=h["relevance_reason"] or "",
                    read=h["arxiv_id"] in already_read,
                )
            )

        return [
            StoredSearchQuery(
                id=r["id"],
                query=r["query"],
                source=r["source"],
                created_at=_format_ts(r["created_at"]),
                session_id=r["session_id"],
                hits=by_query.get(r["id"], []),
            )
            for r in rows
        ]


def _format_ts(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)
