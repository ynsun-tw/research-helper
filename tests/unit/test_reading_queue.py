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
    exec_ingest_local_papers,
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


# ------------------------ local PDF folder ingestion -----------------------


def _make_fake_pdf(folder: Path, name: str, payload: bytes = b"") -> Path:
    """Drop a sham .pdf file so ingest can hash it without real parsing."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    # Distinct content per call keeps sha1s distinct (default name-based seed).
    path.write_bytes(payload or f"%PDF-1.4\n{name}".encode())
    return path


def test_repo_add_persists_pdf_path(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    entry = repo.add(
        "local:abc123", title="hello", pdf_path="/tmp/hello.pdf"
    )
    assert entry.pdf_path == "/tmp/hello.pdf"
    again = repo.get("local:abc123")
    assert again is not None
    assert again.pdf_path == "/tmp/hello.pdf"
    db.close()


def test_repo_add_refreshes_pdf_path_when_changed(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("local:abc123", pdf_path="/old/path.pdf")
    refreshed = repo.add("local:abc123", pdf_path="/new/path.pdf")
    assert refreshed.pdf_path == "/new/path.pdf"
    db.close()


def test_ingest_local_folder_picks_up_pdfs(tmp_path: Path) -> None:
    from research_agent.chat.tools import _ingest_local_folder

    session, _ = _make_session(tmp_path)
    papers = tmp_path / "papers"
    _make_fake_pdf(papers, "attention.pdf")
    _make_fake_pdf(papers, "bert.pdf")
    _make_fake_pdf(papers, "ignored.txt", payload=b"not a pdf")

    result = _ingest_local_folder(session, str(papers), recursive=False)

    assert len(result.added) == 2
    assert not result.refreshed
    assert not result.skipped
    titles = sorted(e.title for e in result.added)
    assert titles == ["attention", "bert"]
    for entry in result.added:
        assert entry.arxiv_id.startswith("local:")
        assert entry.pdf_path.endswith(".pdf")
        assert entry.status == "pending"
    session.close()


def test_ingest_local_folder_is_idempotent(tmp_path: Path) -> None:
    from research_agent.chat.tools import _ingest_local_folder

    session, _ = _make_session(tmp_path)
    papers = tmp_path / "papers"
    _make_fake_pdf(papers, "p1.pdf")

    first = _ingest_local_folder(session, str(papers), recursive=False)
    second = _ingest_local_folder(session, str(papers), recursive=False)

    assert len(first.added) == 1
    assert not second.added
    assert len(second.refreshed) == 1
    assert second.refreshed[0].arxiv_id == first.added[0].arxiv_id
    assert len(session.queue.list()) == 1
    session.close()


def test_ingest_local_folder_recursive_flag(tmp_path: Path) -> None:
    from research_agent.chat.tools import _ingest_local_folder

    session, _ = _make_session(tmp_path)
    root = tmp_path / "library"
    _make_fake_pdf(root, "top.pdf")
    _make_fake_pdf(root / "nested", "deep.pdf")

    shallow = _ingest_local_folder(session, str(root), recursive=False)
    assert len(shallow.added) == 1
    assert shallow.added[0].title == "top"

    # Reset and re-ingest recursively.
    for entry in session.queue.list():
        session.queue.remove(entry.arxiv_id)
    deep = _ingest_local_folder(session, str(root), recursive=True)
    titles = sorted(e.title for e in deep.added)
    assert titles == ["deep", "top"]
    session.close()


def test_ingest_local_folder_reports_missing_dir(tmp_path: Path) -> None:
    from research_agent.chat.tools import _ingest_local_folder

    session, _ = _make_session(tmp_path)
    result = _ingest_local_folder(
        session, str(tmp_path / "nope"), recursive=False
    )
    assert not result.added
    assert result.skipped
    assert "not found" in result.skipped[0][1]
    session.close()


def test_cmd_queue_ingest_subcommand(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    papers = tmp_path / "papers"
    _make_fake_pdf(papers, "alpha.pdf")
    _make_fake_pdf(papers, "beta.pdf")

    cmd_queue(session, f"ingest {papers}")

    out = console.file.getvalue()
    assert "added:" in out
    assert "2" in out
    entries = session.queue.list()
    assert len(entries) == 2
    for e in entries:
        assert e.pdf_path
    session.close()


def test_cmd_queue_ingest_recursive_flag(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    root = tmp_path / "library"
    _make_fake_pdf(root, "top.pdf")
    _make_fake_pdf(root / "nested", "deep.pdf")

    cmd_queue(session, f"ingest {root} --recursive")

    out = console.file.getvalue()
    assert "(recursive)" in out
    assert len(session.queue.list()) == 2
    session.close()


def test_cmd_queue_ingest_usage_when_no_arg(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    cmd_queue(session, "ingest")
    out = console.file.getvalue()
    assert "Usage" in out
    session.close()


def test_exec_ingest_local_papers_returns_summary(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    papers = tmp_path / "papers"
    _make_fake_pdf(papers, "gamma.pdf")

    text = exec_ingest_local_papers(session, {"folder": str(papers)})
    assert "added: 1" in text
    assert "gamma" not in text  # summary doesn't list filenames by default
    assert session.queue.get(session.queue.list()[0].arxiv_id) is not None
    session.close()


def test_exec_ingest_local_papers_missing_folder_arg(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    assert "Error" in exec_ingest_local_papers(session, {})
    assert "Error" in exec_ingest_local_papers(session, {"folder": "   "})
    session.close()


def test_ingest_local_tool_is_registered() -> None:
    assert "ingest_local_papers" in LLM_TOOLS
    schema = LLM_TOOLS["ingest_local_papers"].schema["function"]
    assert "folder" in schema["parameters"]["required"]


def test_cmd_queue_read_uses_pdf_path_for_local_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local entries (no arXiv id) must hand the PDF path to the loader,
    not the content-hash id arXiv can't resolve."""
    session, _ = _make_session(tmp_path)
    papers = tmp_path / "papers"
    pdf = _make_fake_pdf(papers, "solo.pdf")
    session.queue.add(
        "local:fakehash", title="solo", pdf_path=str(pdf)
    )

    captured: dict[str, str] = {}

    def fake_load_and_analyze(s: ChatSession, source: str) -> str:
        captured["source"] = source
        return "ok"

    monkeypatch.setattr(
        "research_agent.chat.tools._load_and_analyze", fake_load_and_analyze
    )
    cmd_queue(session, "read")
    assert captured["source"] == str(pdf)
    session.close()


def test_exec_queue_next_hints_local_pdf_path(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    session.queue.add(
        "local:abc", title="local", pdf_path="/tmp/local.pdf"
    )
    text = exec_queue_next(session, {})
    assert "/tmp/local.pdf" in text
    assert "local PDF" in text
    session.close()


def test_repo_find_by_pdf_path_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    repo = ReadingQueueRepository(db)
    repo.add("local:abc", title="x", pdf_path="/abs/x.pdf")
    repo.add("1706.03762", title="arxiv-only")  # no pdf_path

    hit = repo.find_by_pdf_path("/abs/x.pdf")
    assert hit is not None
    assert hit.arxiv_id == "local:abc"
    assert repo.find_by_pdf_path("/nope") is None
    assert repo.find_by_pdf_path("") is None  # empty path is a no-op
    db.close()


def test_auto_ingest_cwd_adds_new_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_agent.chat.tools import _auto_ingest_cwd

    workdir = tmp_path / "papers"
    workdir.mkdir()
    _make_fake_pdf(workdir, "a.pdf")
    _make_fake_pdf(workdir, "b.pdf")

    session, _ = _make_session(tmp_path)
    monkeypatch.chdir(workdir)
    result = _auto_ingest_cwd(session)
    assert result.added == 2
    assert result.skipped_known == 0
    assert result.folder == workdir
    paths = {e.pdf_path for e in session.queue.list()}
    assert paths == {str((workdir / "a.pdf").resolve()),
                     str((workdir / "b.pdf").resolve())}
    session.close()


def test_auto_ingest_cwd_skips_known_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second startup should NOT re-hash files already in the queue."""
    from research_agent.chat.tools import _auto_ingest_cwd, _hash_pdf

    workdir = tmp_path / "papers"
    workdir.mkdir()
    pdf = _make_fake_pdf(workdir, "a.pdf")

    session, _ = _make_session(tmp_path)
    monkeypatch.chdir(workdir)
    _auto_ingest_cwd(session)

    calls = {"n": 0}
    real_hash = _hash_pdf

    def counting_hash(path: Path) -> str:
        calls["n"] += 1
        return real_hash(path)

    monkeypatch.setattr(
        "research_agent.chat.tools._hash_pdf", counting_hash
    )
    second = _auto_ingest_cwd(session)
    assert second.added == 0
    assert second.skipped_known == 1
    assert calls["n"] == 0  # never hashed again
    assert pdf.exists()  # sanity
    session.close()


def test_auto_ingest_cwd_silent_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_agent.chat.tools import _auto_ingest_cwd

    workdir = tmp_path / "empty"
    workdir.mkdir()
    session, _ = _make_session(tmp_path)
    monkeypatch.chdir(workdir)
    result = _auto_ingest_cwd(session)
    assert result.added == 0
    assert result.skipped_known == 0
    session.close()


def test_auto_ingest_cwd_is_not_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_agent.chat.tools import _auto_ingest_cwd

    workdir = tmp_path / "library"
    workdir.mkdir()
    _make_fake_pdf(workdir, "top.pdf")
    _make_fake_pdf(workdir / "nested", "deep.pdf")

    session, _ = _make_session(tmp_path)
    monkeypatch.chdir(workdir)
    result = _auto_ingest_cwd(session)
    assert result.added == 1
    titles = [e.title for e in session.queue.list()]
    assert titles == ["top"]
    session.close()


def test_render_queue_marks_local_entries(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    session.queue.add("1706.03762", title="Attention")  # arXiv entry
    session.queue.add(
        "local:abc", title="my-paper", pdf_path="/tmp/x.pdf"
    )
    cmd_queue(session, "list all")
    out = console.file.getvalue()
    assert "my-paper" in out
    assert "local" in out  # the (local) badge
    session.close()
