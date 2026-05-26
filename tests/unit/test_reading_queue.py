"""Tests for the reading queue (M3 task 3)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    LLM_TOOLS,
    cmd_queue,
    exec_queue_add,
    exec_queue_list,
    exec_queue_next,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.storage.database import Database
from research_agent.storage.reading_queue import (
    ALLOWED_STATUSES,
    ReadingQueueRepository,
)


def _make_session(tmp_path: Path) -> tuple[ChatSession, Console]:
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=140)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    return session, console


# ----------------------------- repository ---------------------------------


def test_repo_add_creates_pending_entry(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    entry = repo.add("1706.03762", title="Attention Is All You Need")
    assert entry.status == "pending"
    assert entry.title == "Attention Is All You Need"
    assert entry.completed_at is None
    db.close()


def test_repo_add_is_idempotent_and_refreshes_metadata(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("1706.03762")
    repo.add("1706.03762", title="Attention Is All You Need")
    entries = repo.list()
    assert len(entries) == 1
    assert entries[0].title == "Attention Is All You Need"
    db.close()


def test_repo_set_status_to_done_stamps_completed_at(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("1706.03762")
    entry = repo.set_status("1706.03762", "done")
    assert entry is not None
    assert entry.status == "done"
    assert entry.completed_at is not None
    db.close()


def test_repo_set_status_rejects_invalid(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("1706.03762")
    with pytest.raises(ValueError):
        repo.set_status("1706.03762", "archived")  # type: ignore[arg-type]
    db.close()


def test_repo_remove_returns_false_when_missing(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    assert repo.remove("does-not-exist") is False
    db.close()


def test_repo_next_pending_is_fifo_and_skips_done(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("a", title="A")
    repo.add("b", title="B")
    repo.add("c", title="C")
    repo.set_status("a", "done")
    entry = repo.next_pending()
    assert entry is not None
    assert entry.arxiv_id == "b"
    db.close()


def test_repo_list_filters_by_status(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("a")
    repo.add("b")
    repo.set_status("b", "skipped")
    pending = repo.list(status="pending")
    skipped = repo.list(status="skipped")
    assert [e.arxiv_id for e in pending] == ["a"]
    assert [e.arxiv_id for e in skipped] == ["b"]
    db.close()


def test_repo_add_rejects_empty_id(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    with pytest.raises(ValueError):
        repo.add("   ")
    db.close()


# ----------------------------- slash command -------------------------------


def test_cmd_queue_default_lists_pending(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    session.queue.add("1706.03762", title="Attention")
    cmd_queue(session, "")
    out = console.file.getvalue()
    assert "1706.03762" in out
    assert "Attention" in out
    session.close()


def test_cmd_queue_add_and_remove_roundtrip(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    cmd_queue(session, "add 1706.03762 Attention Is All You Need")
    assert session.queue.get("1706.03762") is not None
    cmd_queue(session, "remove 1706.03762")
    assert session.queue.get("1706.03762") is None
    out = console.file.getvalue()
    assert "Queued 1706.03762" in out
    assert "Removed 1706.03762" in out
    session.close()


def test_cmd_queue_done_marks_status(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    session.queue.add("a")
    cmd_queue(session, "done a")
    assert session.queue.get("a").status == "done"  # type: ignore[union-attr]
    session.close()


def test_cmd_queue_skip_marks_status(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    session.queue.add("a")
    cmd_queue(session, "skip a")
    assert session.queue.get("a").status == "skipped"  # type: ignore[union-attr]
    session.close()


def test_cmd_queue_next_shows_first_pending(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    session.queue.add("a", title="First")
    session.queue.add("b", title="Second")
    cmd_queue(session, "next")
    out = console.file.getvalue()
    assert "Next up" in out
    assert "First" in out
    assert "Second" not in out
    session.close()


def test_cmd_queue_next_empty_says_so(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    cmd_queue(session, "next")
    assert "Queue is empty" in console.file.getvalue()
    session.close()


def test_cmd_queue_list_unknown_status_warns(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    cmd_queue(session, "list nonsense")
    assert "Unknown filter" in console.file.getvalue()
    session.close()


def test_cmd_queue_list_all_includes_done(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    session.queue.add("a", title="A")
    session.queue.set_status("a", "done")
    cmd_queue(session, "list all")
    assert "A" in console.file.getvalue()
    session.close()


# -------------------------- /queue read (batch read) -----------------------


def test_cmd_queue_read_pulls_next_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    session.queue.add("1706.03762", title="Attention")
    session.queue.add("1810.04805", title="BERT")

    captured: dict[str, str] = {}

    def fake_load_and_analyze(s: ChatSession, source: str) -> str:
        captured["source"] = source
        # Simulate the real flow's "mark done if queued" effect.
        s.queue.set_status(source, "done")
        return f"Loaded paper '{source}'."

    monkeypatch.setattr(
        "research_agent.chat.tools._load_and_analyze", fake_load_and_analyze
    )
    cmd_queue(session, "read")
    assert captured["source"] == "1706.03762"
    entry = session.queue.get("1706.03762")
    assert entry is not None
    assert entry.status == "done"
    out = console.file.getvalue()
    assert "Reading next pending" in out
    session.close()


def test_cmd_queue_read_empty_says_so(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    cmd_queue(session, "read")
    assert "Queue is empty" in console.file.getvalue()
    session.close()


# ------------------------------ LLM tools ---------------------------------


def test_llm_tools_are_registered() -> None:
    for name in ("queue_add", "queue_list", "queue_next"):
        assert name in LLM_TOOLS
    add_schema = LLM_TOOLS["queue_add"].schema["function"]
    assert "arxiv_id" in add_schema["parameters"]["required"]


def test_exec_queue_add_handles_missing_arxiv_id(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    assert "Error" in exec_queue_add(session, {})
    assert "Error" in exec_queue_add(session, {"arxiv_id": "   "})
    session.close()


def test_exec_queue_add_persists_entry(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    text = exec_queue_add(
        session, {"arxiv_id": "1706.03762", "title": "Attention"}
    )
    assert "Queued 1706.03762" in text
    assert session.queue.get("1706.03762") is not None
    session.close()


def test_exec_queue_list_filters(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    session.queue.add("a", title="A")
    session.queue.add("b", title="B")
    session.queue.set_status("b", "done")

    pending = exec_queue_list(session, {})
    assert "1 entry" in pending
    assert " a " in pending or "- a " in pending

    all_text = exec_queue_list(session, {"status": "all"})
    assert "2 entry/ies" in all_text
    assert "[done]" in all_text

    err = exec_queue_list(session, {"status": "nonsense"})
    assert err.startswith("Error:")
    session.close()


def test_exec_queue_next_returns_arxiv_id(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    assert exec_queue_next(session, {}) == "Queue is empty."
    session.queue.add("1706.03762", title="Attention")
    out = exec_queue_next(session, {})
    assert "1706.03762" in out
    assert "load_paper" in out
    session.close()


# -------------------------- read auto-marks queue --------------------------


def test_load_and_analyze_marks_queued_paper_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When /read loads a paper that's in the queue, mark it as done."""
    from research_agent.chat import tools as chat_tools
    from research_agent.core.paper import Paper, Section

    session, _ = _make_session(tmp_path)
    session.queue.add("1706.03762", title="Attention")

    # Real paper ids are prefixed; this guards against regressing the
    # prefix-strip needed to find the matching queue entry.
    paper = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        authors=["A"],
        abstract="abs",
        sections=[Section(title="Intro", content="hi")],
    )

    monkeypatch.setattr(chat_tools, "_load_anchor_paper", lambda *a, **k: paper)

    from research_agent.agents.analyst import AnalysisResult
    from research_agent.agents.critic import CritiqueResult
    from research_agent.agents.orchestrator import AggregatedAnalysis

    analyst = AnalysisResult(
        contributions=["c"],
        method_insights=[],
        potential_impact="",
        related_work=[],
        claimed_vs_evidence=[],
        confidence=0.5,
    )
    critic = CritiqueResult(
        objections=[],
        support_score=7.0,
        score_reason="ok",
    )
    report = AggregatedAnalysis(analyst=analyst, critic=critic)

    async def fake_analyze(_paper):
        return report

    session.orch.analyze_paper_parallel = fake_analyze  # type: ignore[assignment]
    monkeypatch.setattr(chat_tools, "render_paper_header", lambda *a, **k: None)
    monkeypatch.setattr(chat_tools, "render_read_report", lambda *a, **k: None)

    chat_tools._load_and_analyze(session, "1706.03762")

    entry = session.queue.get("1706.03762")
    assert entry is not None
    assert entry.status == "done"
    session.close()


def test_all_allowed_statuses_constant() -> None:
    assert {"pending", "in_progress", "done", "skipped"} == ALLOWED_STATUSES
