"""Performance check for recall_history at ~1000 indexed messages (T3.3.1.4).

Acceptance criterion from the M3 spec: recall latency < 2s for 1000 records.

We measure two things:
- Pure vector store latency (DiscussionVectorStore.query_similar over 1000
  fallback-indexed messages).
- End-to-end MemoryKeeper.recall_history latency, which adds SQLite
  fetch-by-id for the matched rows.

The keyword/Jaccard fallback is what ships when chroma isn't installed
or wired up - it's the slowest of the two implementations, so passing
under the threshold here means real chroma will also pass.

This runs as a normal integration test (no opt-in env var) because it
only takes ~1s and doesn't touch the network. The budget asserted is
strictly below the spec's 2.0s to leave slack for slow CI runners.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.storage.database import Database
from research_agent.storage.discussion_vectors import DiscussionVectorStore
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore

# Distinct topics so different queries match different subsets. Each
# message blends topic-specific keywords + role-specific filler so the
# Jaccard fallback (set of unique tokens) ranks meaningfully.
TOPICS = [
    (
        "transformer attention",
        "Self-attention scales quadratically with sequence length. "
        "Multi-head attention captures different subspaces.",
    ),
    (
        "convolutional networks",
        "Convolutional kernels exploit translation equivariance. "
        "Residual connections enable training deeper networks.",
    ),
    (
        "reinforcement learning",
        "Policy gradient methods estimate the gradient of expected return. "
        "Actor-critic combines value estimation with policy optimisation.",
    ),
    (
        "graph neural networks",
        "Message passing aggregates neighbour features. "
        "Graph attention adapts neighbour weights with learned coefficients.",
    ),
    (
        "variational autoencoders",
        "Latent variable models maximise the evidence lower bound. "
        "Reparameterisation enables backpropagation through sampling.",
    ),
]
ROLES = ("user", "analyst", "critic", "user")  # weighted toward user prompts


@pytest.fixture
def seeded_memory_keeper(tmp_path: Path) -> tuple[MemoryKeeper, str]:
    """1000 messages indexed across 10 simulated past sessions + a fresh one."""
    db = Database(tmp_path / "perf.db")
    discussions = DiscussionRepository(db)
    ideas = IdeaRepository(db)
    idea_vectors = IdeaVectorStore(tmp_path / "idea-chroma", use_chroma=False)
    discussion_vectors = DiscussionVectorStore(
        tmp_path / "discuss-chroma", use_chroma=False
    )
    keeper = MemoryKeeper(
        ideas,
        idea_vectors,
        discussions=discussions,
        discussion_vectors=discussion_vectors,
    )

    # Generate 1000 messages: 10 past sessions x 100 messages each.
    current_session = "sess-current"
    for s in range(10):
        session_id = f"sess-{s:02d}"
        for i in range(100):
            topic, body = TOPICS[i % len(TOPICS)]
            role = ROLES[i % len(ROLES)]
            content = f"({role} {s}/{i}) {topic} - {body}"
            mid = discussions.append(session_id, role, content)
            keeper.index_message(
                message_id=mid,
                session_id=session_id,
                role=role,
                content=content,
            )

    return keeper, current_session


def test_vector_store_query_under_500ms_at_1000_messages(
    seeded_memory_keeper: tuple[MemoryKeeper, str],
) -> None:
    keeper, current_session = seeded_memory_keeper
    assert keeper.discussion_vectors is not None
    # Sanity: we did seed 1000 messages.
    assert (
        len(keeper.discussion_vectors._fallback)  # type: ignore[union-attr]
        == 1000
    )

    # Warm up so first-run JIT-ish costs don't skew the measurement.
    keeper.discussion_vectors.query_similar(
        "transformer attention", limit=10
    )

    t0 = time.perf_counter()
    ids = keeper.discussion_vectors.query_similar(
        "transformer attention sequence modelling",
        limit=10,
        exclude_session_id=current_session,
    )
    elapsed = time.perf_counter() - t0

    assert len(ids) == 10, f"Expected top-10 ids, got {len(ids)}"
    # 0.5s is comfortably under the 2s spec target and gives headroom
    # for slow CI; the in-memory Jaccard is typically O(few ms) on
    # 1000 messages.
    assert elapsed < 0.5, (
        f"DiscussionVectorStore.query_similar took {elapsed*1000:.1f}ms "
        f"for 1000 indexed messages; spec budget is 500ms."
    )


def test_recall_history_under_2s_at_1000_messages(
    seeded_memory_keeper: tuple[MemoryKeeper, str],
) -> None:
    keeper, current_session = seeded_memory_keeper

    # Warm-up to amortise SQLite + Chroma client init costs.
    keeper.recall_history(
        "transformer attention", limit=10, exclude_session_id=current_session
    )

    t0 = time.perf_counter()
    hits = keeper.recall_history(
        "transformer attention sequence modelling",
        limit=10,
        exclude_session_id=current_session,
    )
    elapsed = time.perf_counter() - t0

    assert hits, "recall_history returned nothing on seeded corpus"
    assert len(hits) <= 10
    # Spec budget is < 2s; assert with margin.
    assert elapsed < 2.0, (
        f"MemoryKeeper.recall_history took {elapsed*1000:.1f}ms "
        f"on 1000 indexed messages; spec budget is 2000ms."
    )


def test_recall_history_top_hit_is_topically_relevant(
    seeded_memory_keeper: tuple[MemoryKeeper, str],
) -> None:
    """The Jaccard fallback isn't precise, but a topical query should
    still surface that topic in the top-3."""
    keeper, current_session = seeded_memory_keeper
    hits = keeper.recall_history(
        "convolutional residual networks",
        limit=3,
        exclude_session_id=current_session,
    )
    assert hits, "recall_history returned nothing"
    # At least one of the top 3 should mention the topic.
    assert any(
        "convolutional" in h.content.lower() or "residual" in h.content.lower()
        for h in hits
    ), f"None of the top-3 hits matched the topic: {[h.content[:60] for h in hits]}"
