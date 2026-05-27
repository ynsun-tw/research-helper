"""Memory Keeper similarity recall (keyword fallback)."""

from __future__ import annotations

import pytest

from research_agent.agents.memory_keeper import (
    DEFAULT_ASSOCIATION_STATUSES,
    DEFAULT_ASSOCIATION_THRESHOLD,
    Association,
    MemoryKeeper,
)
from research_agent.storage.database import Database
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore


def _make_keeper(config_dir):
    db = Database(config_dir / "memory.db")
    ideas = IdeaRepository(db)
    vectors = IdeaVectorStore(config_dir / "chroma", use_chroma=False)
    keeper = MemoryKeeper(ideas, vectors)
    return db, ideas, vectors, keeper


def test_recall_similar_idea(config_dir) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    a = ideas.create("Transformer caching", "Speed up inference with KV cache reuse")
    b = ideas.create("Gardening tips", "How to grow tomatoes in winter")
    vectors.upsert(a)
    vectors.upsert(b)

    similar = keeper.recall_similar("KV cache for faster transformer inference", limit=2)
    assert similar
    assert similar[0].id == a.id
    db.close()


# ---------------------- check_associations (T3.4.1.1) -------------------


def _create_and_set_status(ideas, vectors, title, description, status):
    """Helper - create + persist + transition + re-upsert in one call."""
    idea = ideas.create(title, description)
    if status != "active":
        ideas.update_status(idea.id, status)
    idea = ideas.get(idea.id)
    assert idea is not None
    vectors.upsert(idea)
    return idea


def test_check_associations_returns_shelved_idea_above_threshold(
    config_dir,
) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    shelved = _create_and_set_status(
        ideas,
        vectors,
        "KV cache reuse for transformer inference",
        "shelved because no GPU budget at the time; would speed transformer "
        "inference via KV cache reuse",
        "shelved",
    )

    # Threshold deliberately low for the keyword Jaccard backend (real
    # production uses chroma with the 0.8 default; Jaccard scores are
    # smaller numerically for the same conceptual match).
    associations = keeper.check_associations(
        "transformer inference KV cache reuse",
        threshold=0.2,
    )
    assert associations
    assert associations[0].idea.id == shelved.id
    assert 0.0 <= associations[0].similarity <= 1.0
    db.close()


def test_check_associations_filters_below_threshold(config_dir) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    _create_and_set_status(
        ideas,
        vectors,
        "Gardening tips",
        "How to grow tomatoes in winter, hardly related to transformers",
        "shelved",
    )

    # Use the production default (0.8) - the weakly related idea must not
    # cross it for this query.
    associations = keeper.check_associations(
        "transformer inference KV cache reuse",
        threshold=DEFAULT_ASSOCIATION_THRESHOLD,
    )
    assert associations == []
    db.close()


def test_check_associations_skips_active_ideas_by_default(config_dir) -> None:
    """The user is already aware of active ideas - don't re-alert them."""
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    _create_and_set_status(
        ideas,
        vectors,
        "Transformer caching",
        "KV cache reuse for transformer inference, currently in progress",
        "active",
    )
    associations = keeper.check_associations(
        "transformer inference KV cache reuse", threshold=0.2
    )
    assert associations == []
    db.close()


def test_check_associations_includes_waiting_status(config_dir) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    _create_and_set_status(
        ideas,
        vectors,
        "KV cache reuse",
        "transformer inference KV cache reuse - waiting for blockers",
        "waiting",
    )
    associations = keeper.check_associations(
        "transformer inference KV cache reuse", threshold=0.2
    )
    assert associations
    assert associations[0].idea.status == "waiting"
    db.close()


def test_check_associations_explicit_status_filter(config_dir) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    _create_and_set_status(
        ideas,
        vectors,
        "abandoned KV cache work",
        "transformer inference KV cache reuse - abandoned long ago",
        "abandoned",
    )
    # Default filter shouldn't include abandoned.
    default = keeper.check_associations(
        "transformer inference KV cache reuse", threshold=0.2
    )
    assert default == []
    # Caller can broaden the filter.
    broader = keeper.check_associations(
        "transformer inference KV cache reuse",
        threshold=0.2,
        statuses=("abandoned",),
    )
    assert broader and broader[0].idea.status == "abandoned"
    db.close()


def test_check_associations_skips_orphan_vector_rows(config_dir) -> None:
    """If a vector outlives its SQLite row, the helper must skip it
    instead of crashing."""
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    shelved = _create_and_set_status(
        ideas,
        vectors,
        "KV cache",
        "transformer inference KV cache reuse",
        "shelved",
    )
    # Simulate an orphaned vector by deleting the SQL row directly.
    with db.conn:
        db.conn.execute("DELETE FROM ideas WHERE id = ?", (shelved.id,))
    associations = keeper.check_associations(
        "transformer inference KV cache reuse", threshold=0.2
    )
    assert associations == []
    db.close()


def test_check_associations_respects_limit(config_dir) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    for i in range(8):
        _create_and_set_status(
            ideas,
            vectors,
            f"shelved idea {i}",
            f"transformer inference KV cache reuse variant {i}",
            "shelved",
        )
    associations = keeper.check_associations(
        "transformer inference KV cache reuse", threshold=0.2, limit=3
    )
    assert len(associations) == 3
    # Sorted by similarity descending.
    for prev, curr in zip(associations, associations[1:]):
        assert prev.similarity >= curr.similarity
    db.close()


def test_check_associations_empty_context_returns_empty(config_dir) -> None:
    _db, _ideas, _vectors, keeper = _make_keeper(config_dir)
    assert keeper.check_associations("") == []
    assert keeper.check_associations("   ") == []
    _db.close()


# --------------------- format_associations rendering -----------------


def test_format_associations_empty_returns_empty_string(config_dir) -> None:
    _db, _ideas, _vectors, keeper = _make_keeper(config_dir)
    assert keeper.format_associations([]) == ""
    _db.close()


def test_format_associations_renders_one_line_per_idea(config_dir) -> None:
    db, ideas, vectors, keeper = _make_keeper(config_dir)
    a = _create_and_set_status(
        ideas, vectors, "KV cache", "KV cache reuse", "shelved"
    )
    rendered = keeper.format_associations(
        [Association(idea=a, similarity=0.91)]
    )
    assert "Related ideas you parked previously" in rendered
    assert "KV cache" in rendered
    assert "91%" in rendered
    assert "shelved" in rendered
    db.close()


# --------------------- module-level defaults / contract --------------


def test_default_threshold_matches_spec() -> None:
    """M3 spec S3.4.1 calls out similarity > 0.8 - the default constant
    is the contract we expose to Config."""
    assert pytest.approx(0.8) == DEFAULT_ASSOCIATION_THRESHOLD


def test_default_statuses_excludes_active() -> None:
    assert "active" not in DEFAULT_ASSOCIATION_STATUSES
    assert "shelved" in DEFAULT_ASSOCIATION_STATUSES


# --------- IdeaVectorStore.query_with_scores contract ---------------


def test_query_with_scores_returns_floats_in_unit_interval(config_dir) -> None:
    db, ideas, vectors, _keeper = _make_keeper(config_dir)
    a = ideas.create("Transformer caching", "KV cache reuse for inference")
    vectors.upsert(a)
    pairs = vectors.query_with_scores("transformer inference cache", limit=3)
    assert pairs, "expected at least one match"
    for _iid, score in pairs:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    db.close()


def test_query_with_scores_orders_by_descending_similarity(config_dir) -> None:
    db, ideas, vectors, _keeper = _make_keeper(config_dir)
    close_match = ideas.create(
        "Transformer caching", "KV cache reuse for transformer inference"
    )
    far_match = ideas.create(
        "Distant idea", "transformer mention but mostly unrelated topic"
    )
    vectors.upsert(close_match)
    vectors.upsert(far_match)
    pairs = vectors.query_with_scores(
        "transformer inference KV cache reuse", limit=2
    )
    assert len(pairs) == 2
    assert pairs[0][1] >= pairs[1][1]
    assert pairs[0][0] == close_match.id
    db.close()


def test_query_similar_still_returns_ids_only(config_dir) -> None:
    """Public surface that didn't ask for scores must keep returning ids."""
    db, ideas, vectors, _keeper = _make_keeper(config_dir)
    a = ideas.create("Transformer caching", "KV cache reuse")
    vectors.upsert(a)
    ids = vectors.query_similar("transformer caching")
    assert ids == [a.id]
    db.close()
