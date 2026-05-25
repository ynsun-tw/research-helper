"""Memory Keeper similarity recall (keyword fallback)."""

from __future__ import annotations

from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.storage.database import Database
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore


def test_recall_similar_idea(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    ideas = IdeaRepository(db)
    vectors = IdeaVectorStore(config_dir / "chroma", use_chroma=False)

    a = ideas.create("Transformer caching", "Speed up inference with KV cache reuse")
    b = ideas.create("Gardening tips", "How to grow tomatoes in winter")
    vectors.upsert(a)
    vectors.upsert(b)

    keeper = MemoryKeeper(ideas, vectors)
    similar = keeper.recall_similar("KV cache for faster transformer inference", limit=2)
    assert similar
    assert similar[0].id == a.id
    db.close()
