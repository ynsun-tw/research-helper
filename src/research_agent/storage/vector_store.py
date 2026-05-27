"""Idea embedding store (ChromaDB with keyword fallback for tests)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from research_agent.core.idea import Idea


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


class IdeaVectorStore:
    """Persist and query Idea descriptions by similarity."""

    def __init__(self, persist_dir: Path, *, use_chroma: bool = True) -> None:
        self.persist_dir = persist_dir
        self._use_chroma = use_chroma
        self._collection: Any = None
        self._fallback: dict[str, tuple[str, str]] = {}

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if not self._use_chroma:
            return self
        import chromadb  # lazy import

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = client.get_or_create_collection(
            name="ideas",
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def upsert(self, idea: Idea) -> None:
        text = f"{idea.title}\n{idea.description}".strip()
        if self._use_chroma:
            coll = self._ensure_collection()
            coll.upsert(
                ids=[idea.id],
                documents=[text],
                metadatas=[{"title": idea.title, "status": idea.status}],
            )
        else:
            self._fallback[idea.id] = (idea.title, text)

    def delete(self, idea_id: str) -> None:
        if self._use_chroma:
            coll = self._ensure_collection()
            if coll is not self:
                coll.delete(ids=[idea_id])
        self._fallback.pop(idea_id, None)

    def query_similar(
        self,
        text: str,
        *,
        limit: int = 3,
        exclude_id: str | None = None,
    ) -> list[str]:
        """Return idea ids ranked by similarity to ``text``."""
        return [
            iid
            for iid, _score in self.query_with_scores(
                text, limit=limit, exclude_id=exclude_id
            )
        ]

    def query_with_scores(
        self,
        text: str,
        *,
        limit: int = 5,
        exclude_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(idea_id, similarity)`` pairs ranked by similarity.

        ``similarity`` is normalised to ``[0.0, 1.0]`` (higher == closer)
        regardless of backend - Chroma's cosine *distance* is mapped to
        ``1.0 - distance`` and the keyword fallback already emits a
        Jaccard ratio in that range. Pairs are sorted descending. The
        threshold-based callers (MemoryKeeper.check_associations) rely
        on this uniform scoring.
        """
        if self._use_chroma:
            coll = self._ensure_collection()
            if coll is not self:
                n = max(limit + (1 if exclude_id else 0), limit)
                result = coll.query(
                    query_texts=[text], n_results=n
                )
                ids: list[str] = list(result.get("ids", [[]])[0])
                distances: list[float] = list(
                    result.get("distances", [[]])[0] or []
                )
                pairs = [
                    (iid, 1.0 - float(dist))
                    for iid, dist in zip(ids, distances)
                ]
                if exclude_id:
                    pairs = [(i, s) for i, s in pairs if i != exclude_id]
                # Clamp to [0,1] in case of numerical edge cases.
                pairs = [(i, max(0.0, min(1.0, s))) for i, s in pairs]
                return pairs[:limit]
        return self._fallback_query_with_scores(
            text, limit=limit, exclude_id=exclude_id
        )

    def _fallback_query_with_scores(
        self,
        text: str,
        *,
        limit: int,
        exclude_id: str | None,
    ) -> list[tuple[str, float]]:
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for iid, (_title, doc) in self._fallback.items():
            if exclude_id and iid == exclude_id:
                continue
            doc_tokens = _tokenize(doc)
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            if overlap > 0:
                scored.append((overlap, iid))
        scored.sort(reverse=True)
        return [(iid, score) for score, iid in scored[:limit]]
