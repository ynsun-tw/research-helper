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
