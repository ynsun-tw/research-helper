"""Unit tests for discussion persistence."""

from __future__ import annotations

from pathlib import Path

from research_agent.storage.database import Database
from research_agent.storage.discussions import DiscussionRepository


def test_append_and_list_session(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = DiscussionRepository(db)
    sid = "sess-1"
    repo.append(sid, "user", "Hello")
    repo.append(sid, "analyst", "Supportive view")
    repo.append(sid, "critic", "Some concerns")
    messages = repo.list_session(sid)
    assert [m.role for m in messages] == ["user", "analyst", "critic"]
    assert messages[0].content == "Hello"
    db.close()
