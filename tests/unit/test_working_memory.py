"""Unit tests for WorkingMemory."""

from __future__ import annotations

from pathlib import Path

from research_agent.memory.working_memory import (
    WorkingMemory,
    estimate_tokens,
)
from research_agent.storage.database import Database
from research_agent.storage.discussions import DiscussionRepository


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_to_context_empty() -> None:
    mem = WorkingMemory.new_session()
    assert mem.to_context(100) == ""


def test_to_context_includes_all_when_small() -> None:
    mem = WorkingMemory.new_session()
    mem.append("user", "hello")
    mem.append("analyst", "hi there")
    ctx = mem.to_context(1000)
    assert "user: hello" in ctx
    assert "analyst: hi there" in ctx


def test_to_context_truncates_oldest_messages() -> None:
    mem = WorkingMemory.new_session()
    mem.append("user", "FIRST" * 200)
    mem.append("analyst", "middle")
    mem.append("user", "LAST")
    # Budget fits only the latest exchange
    ctx = mem.to_context(max_tokens=estimate_tokens("user: LAST\nanalyst: middle") + 5)
    assert "LAST" in ctx
    assert "FIRST" not in ctx


def test_to_context_zero_budget_returns_empty() -> None:
    mem = WorkingMemory.new_session()
    mem.append("user", "x")
    assert mem.to_context(0) == ""


def test_persist_writes_new_messages_only(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = DiscussionRepository(db)
    mem = WorkingMemory.new_session()
    mem.append("user", "one")
    mem.append("analyst", "two")
    n = mem.persist(repo)
    assert n == 2
    assert mem.persist(repo) == 0
    rows = repo.list_session(mem.session_id)
    assert len(rows) == 2
    assert rows[0].role == "user"
    db.close()


def test_from_session_restores_messages(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = DiscussionRepository(db)
    sid = "sess-restore"
    repo.append(sid, "user", "question")
    repo.append(sid, "critic", "pushback")
    mem = WorkingMemory.from_session(repo, sid)
    assert mem.turn_count() == 1
    assert len(mem.messages) == 2
    assert mem.to_context(500).startswith("user:")
    db.close()


def test_turn_count() -> None:
    mem = WorkingMemory.new_session()
    mem.append("user", "a")
    mem.append("analyst", "b")
    mem.append("user", "c")
    assert mem.turn_count() == 2
