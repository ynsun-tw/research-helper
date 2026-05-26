"""Unit tests for Idea repository and status machine."""

from __future__ import annotations

import pytest

from research_agent.core.idea import InvalidStatusTransition, can_transition
from research_agent.storage.database import Database
from research_agent.storage.ideas import IdeaRepository


def test_status_transitions() -> None:
    assert can_transition("active", "shelved")
    assert not can_transition("completed", "shelved")


def test_idea_crud_and_score_history(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = IdeaRepository(db)
    idea = repo.create("Title", "Description of the idea")
    loaded = repo.get(idea.id)
    assert loaded is not None
    assert loaded.title == "Title"

    repo.append_score(idea.id, 6.0, "Needs work", session_id="sess-1")
    updated = repo.get(idea.id)
    assert updated is not None
    assert len(updated.score_history) == 1
    assert updated.critic_score == 6.0

    repo.add_user_score_feedback(idea.id, "Score feels high")
    with_feedback = repo.get(idea.id)
    assert with_feedback is not None
    assert "Score feels high" in with_feedback.user_score_feedback
    db.close()


def test_invalid_status_transition(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = IdeaRepository(db)
    idea = repo.create("X", status="completed")
    with pytest.raises(InvalidStatusTransition):
        repo.update_status(idea.id, "shelved")
    db.close()


# ---------------------------------------------------------------------------
# M3 T3.4.2 — activation conditions on shelved/waiting ideas
# ---------------------------------------------------------------------------


def test_activation_conditions_persist(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = IdeaRepository(db)
    idea = repo.create("Train a small SLM on FineWeb-Edu")
    repo.add_activation_condition(idea.id, "FineWeb-Edu dataset release")
    repo.add_activation_condition(idea.id, "1B model checkpoint")
    loaded = repo.get(idea.id)
    assert loaded is not None
    assert loaded.activation_conditions == [
        "FineWeb-Edu dataset release",
        "1B model checkpoint",
    ]
    db.close()


def test_activation_conditions_dedupe_case_insensitive(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = IdeaRepository(db)
    idea = repo.create("Idea")
    repo.add_activation_condition(idea.id, "FineWeb dataset")
    repo.add_activation_condition(idea.id, "fineweb DATASET")  # dup
    repo.add_activation_condition(idea.id, "  ")  # empty -> noop
    loaded = repo.get(idea.id)
    assert loaded is not None
    assert loaded.activation_conditions == ["FineWeb dataset"]
    db.close()


def test_clear_activation_conditions(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = IdeaRepository(db)
    idea = repo.create("Idea")
    repo.add_activation_condition(idea.id, "needs xxx")
    repo.clear_activation_conditions(idea.id)
    loaded = repo.get(idea.id)
    assert loaded is not None
    assert loaded.activation_conditions == []
    db.close()


def test_activation_conditions_migrate_legacy_row(config_dir) -> None:
    """A row written before T3.4.2 (no activation_conditions column value)
    should round-trip as an empty list rather than crash."""
    db = Database(config_dir / "memory.db")
    repo = IdeaRepository(db)
    idea = repo.create("Legacy")
    # Simulate older write that left the column NULL.
    db.conn.execute(
        "UPDATE ideas SET activation_conditions = NULL WHERE id = ?",
        (idea.id,),
    )
    db.conn.commit()
    loaded = repo.get(idea.id)
    assert loaded is not None
    assert loaded.activation_conditions == []
    db.close()
