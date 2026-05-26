"""Vector store for cross-session discussion recall (M3 task 2).

Indexes individual ``DiscussionMessage`` rows (user prompts + Analyst /
Critic conclusions) so MemoryKeeper can answer "have we talked about X
before?" across past REPL sessions.

Mirrors :class:`IdeaVectorStore`: ChromaDB persistent collection in
production, in-memory keyword Jaccard fallback for tests where chroma is
disabled.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


# Roles whose content is worth recalling later. Excludes "system"/"tool"
# (noise from slash output + tool results) and the orchestrator.
INDEXABLE_ROLES = frozenset({"user", "analyst", "critic"})


class DiscussionVectorStore:
    """Persist and query DiscussionMessage texts by similarity."""

    def __init__(self, persist_dir: Path, *, use_chroma: bool = True) -> None:
        self.persist_dir = persist_dir
        self._use_chroma = use_chroma
        self._collection: Any = None
        # id → (role, session_id, content)
        self._fallback: dict[str, tuple[str, str, str]] = {}

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if not self._use_chroma:
            return self
        import chromadb  # lazy import

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = client.get_or_create_collection(
            name="discussions",
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def upsert(
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        idea_id: str | None = None,
    ) -> None:
        text = content.strip()
        if not text:
            return
        meta: dict[str, Any] = {"role": role, "session_id": session_id}
        if idea_id:
            meta["idea_id"] = idea_id
        if self._use_chroma:
            coll = self._ensure_collection()
            if coll is not self:
                coll.upsert(
                    ids=[message_id],
                    documents=[text],
                    metadatas=[meta],
                )
                return
        self._fallback[message_id] = (role, session_id, text)

    def query_similar(
        self,
        text: str,
        *,
        limit: int = 5,
        exclude_session_id: str | None = None,
    ) -> list[str]:
        if self._use_chroma:
            coll = self._ensure_collection()
            if coll is not self:
                n = max(limit + (10 if exclude_session_id else 0), limit)
                result = coll.query(query_texts=[text], n_results=n)
                ids: list[str] = list(result.get("ids", [[]])[0])
                metadatas: list[dict[str, Any]] = list(
                    result.get("metadatas", [[]])[0]
                )
                if exclude_session_id:
                    paired = list(zip(ids, metadatas, strict=False))
                    ids = [
                        mid
                        for mid, meta in paired
                        if (meta or {}).get("session_id") != exclude_session_id
                    ]
                return ids[:limit]
        return self._fallback_query(
            text, limit=limit, exclude_session_id=exclude_session_id
        )

    def _fallback_query(
        self,
        text: str,
        *,
        limit: int,
        exclude_session_id: str | None,
    ) -> list[str]:
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for mid, (_role, session_id, doc) in self._fallback.items():
            if exclude_session_id and session_id == exclude_session_id:
                continue
            doc_tokens = _tokenize(doc)
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            if overlap > 0:
                scored.append((overlap, mid))
        scored.sort(reverse=True)
        return [mid for _, mid in scored[:limit]]
