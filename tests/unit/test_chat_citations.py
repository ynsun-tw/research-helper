"""Tests for /cites, /refs slash commands and get_citations / get_references
LLM tools (M3 T3.1.2.2)."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    LLM_TOOLS,
    SLASH_COMMANDS,
    _resolve_citation_target,
    cmd_cites,
    cmd_refs,
    exec_get_citations,
    exec_get_references,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper
from research_agent.search.arxiv_search import ArxivSearchError, ArxivSearchHit


def _make_session(tmp_path) -> tuple[ChatSession, Console]:
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    return session, console


def _hit(arxiv_id: str, title: str, year: str = "2018") -> ArxivSearchHit:
    return ArxivSearchHit(
        arxiv_id=arxiv_id,
        title=title,
        abstract=f"Abstract for {title}",
        published=f"{year}-01-01",
        source="semantic_scholar",
    )


# ----------------------------- registration --------------------------


def test_slash_and_llm_tools_registered() -> None:
    assert "cites" in SLASH_COMMANDS
    assert "refs" in SLASH_COMMANDS
    assert "get_citations" in LLM_TOOLS
    assert "get_references" in LLM_TOOLS
    schema = LLM_TOOLS["get_citations"].schema["function"]
    assert schema["parameters"]["required"] == ["arxiv_id"]


# ---------------------------- target resolver -------------------------


def test_resolve_target_uses_explicit_argument(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    try:
        assert _resolve_citation_target(session, "1706.03762") == "1706.03762"
    finally:
        session.close()


def test_resolve_target_falls_back_to_anchor_paper(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    session.anchor_paper = Paper(
        id="arxiv:1706.03762", title="Attention", abstract="", full_text=""
    )
    try:
        assert _resolve_citation_target(session, "") == "1706.03762"
    finally:
        session.close()


def test_resolve_target_rejects_local_anchor(tmp_path) -> None:
    """Local PDFs have no arXiv mapping -> citation graph unavailable."""
    session, _ = _make_session(tmp_path)
    session.anchor_paper = Paper(
        id="local:abc123", title="Local PDF", abstract="", full_text=""
    )
    try:
        assert _resolve_citation_target(session, "") is None
    finally:
        session.close()


def test_resolve_target_returns_none_without_anchor_or_arg(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    try:
        assert _resolve_citation_target(session, "") is None
    finally:
        session.close()


# ----------------------- /cites slash handler -------------------------


def test_cmd_cites_without_target_prints_usage(tmp_path) -> None:
    session, console = _make_session(tmp_path)
    try:
        cmd_cites(session, "")
        out = console.file.getvalue()
        assert "Usage" in out
        assert "/cites" in out
    finally:
        session.close()


def test_cmd_cites_calls_s2_and_renders_table(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    hits = [
        _hit("2001.00001", "Follow-up paper", "2020"),
        _hit("2010.10000", "Another follow-up", "2020"),
    ]
    captured: dict[str, object] = {}

    def fake_get_citations(self, arxiv_id, *, max_results=10):  # type: ignore[no-untyped-def]
        captured["arxiv_id"] = arxiv_id
        captured["max_results"] = max_results
        return hits

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_citations",
        fake_get_citations,
    )
    try:
        cmd_cites(session, "1706.03762")
        out = console.file.getvalue()
        assert captured["arxiv_id"] == "1706.03762"
        assert captured["max_results"] == 10
        assert "Citations of arxiv:1706.03762" in out
        assert "2001.00001" in out
        assert "Follow-up paper" in out
        # Working memory should record the summary.
        sys_msgs = [m for m in session.memory.messages if m.role == "system"]
        assert any("citations" in m.content for m in sys_msgs)
    finally:
        session.close()


def test_cmd_cites_handles_empty_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_citations",
        lambda self, arxiv_id, *, max_results=10: [],
    )
    try:
        cmd_cites(session, "9999.99999")
        out = console.file.getvalue()
        assert "No citations found" in out
    finally:
        session.close()


def test_cmd_cites_handles_arxiv_search_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)

    def boom(self, arxiv_id, *, max_results=10):  # type: ignore[no-untyped-def]
        raise ArxivSearchError("HTTP 429")

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_citations",
        boom,
    )
    try:
        cmd_cites(session, "1706.03762")
        out = console.file.getvalue()
        assert "Error" in out and "429" in out
    finally:
        session.close()


def test_cmd_cites_uses_anchor_paper_when_no_arg(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _make_session(tmp_path)
    session.anchor_paper = Paper(
        id="arxiv:1706.03762", title="Attention", abstract="", full_text=""
    )
    captured: dict[str, str] = {}

    def fake_get_citations(self, arxiv_id, *, max_results=10):  # type: ignore[no-untyped-def]
        captured["arxiv_id"] = arxiv_id
        return [_hit("2001.00001", "Cited paper")]

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_citations",
        fake_get_citations,
    )
    try:
        cmd_cites(session, "")
        assert captured["arxiv_id"] == "1706.03762"
    finally:
        session.close()


# ----------------------- /refs slash handler --------------------------


def test_cmd_refs_calls_get_references(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    captured: dict[str, str] = {}

    def fake_get_references(self, arxiv_id, *, max_results=10):  # type: ignore[no-untyped-def]
        captured["arxiv_id"] = arxiv_id
        return [_hit("1409.0473", "Foundational")]

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_references",
        fake_get_references,
    )
    try:
        cmd_refs(session, "1706.03762")
        out = console.file.getvalue()
        assert captured["arxiv_id"] == "1706.03762"
        assert "References from arxiv:1706.03762" in out
        assert "1409.0473" in out
        assert "Foundational" in out
    finally:
        session.close()


# ----------------------- LLM-tool executors ---------------------------


def test_exec_get_citations_returns_summary_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _make_session(tmp_path)
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_citations",
        lambda self, arxiv_id, *, max_results=10: [
            _hit("2001.00001", "Follow-up"),
            _hit("2010.10000", "Another follow-up"),
        ],
    )
    try:
        out = exec_get_citations(
            session, {"arxiv_id": "1706.03762", "max_results": 5}
        )
        assert "Found 2 citations" in out
        assert "2001.00001" in out
        assert "2010.10000" in out
    finally:
        session.close()


def test_exec_get_citations_validates_arxiv_id(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    try:
        assert exec_get_citations(session, {"arxiv_id": ""}).startswith("Error")
        assert exec_get_citations(session, {}).startswith("Error")
    finally:
        session.close()


def test_exec_get_citations_clamps_max_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _make_session(tmp_path)
    captured: dict[str, int] = {}

    def fake_get_citations(self, arxiv_id, *, max_results=10):  # type: ignore[no-untyped-def]
        captured["max_results"] = max_results
        return []

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_citations",
        fake_get_citations,
    )
    try:
        exec_get_citations(session, {"arxiv_id": "x", "max_results": 1000})
        assert captured["max_results"] == 25  # clamped to schema max
        exec_get_citations(session, {"arxiv_id": "x", "max_results": 0})
        assert captured["max_results"] == 1  # clamped to schema min
    finally:
        session.close()


def test_exec_get_references_marks_already_read_papers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _make_session(tmp_path)
    # Seed library with one of the references we'll surface.
    session.papers.save(
        Paper(id="arxiv:1409.0473", title="Foundational", abstract="", full_text="")
    )
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_references",
        lambda self, arxiv_id, *, max_results=10: [
            _hit("1409.0473", "Foundational"),
            _hit("9999.99999", "Unread paper"),
        ],
    )
    try:
        out = exec_get_references(session, {"arxiv_id": "1706.03762"})
        assert "1409.0473" in out
        assert "[read]" in out
        assert "9999.99999" in out
    finally:
        session.close()


def test_exec_get_references_returns_error_on_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _make_session(tmp_path)

    def boom(self, arxiv_id, *, max_results=10):  # type: ignore[no-untyped-def]
        raise ArxivSearchError("upstream down")

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.SemanticScholarSearcher.get_references",
        boom,
    )
    try:
        out = exec_get_references(session, {"arxiv_id": "1706.03762"})
        assert out.startswith("Error:")
    finally:
        session.close()


# ------------------------ system prompt mentions ---------------------


def test_system_prompt_advertises_citation_tools() -> None:
    from research_agent.chat.router import CHAT_SYSTEM_PROMPT

    assert "get_citations" in CHAT_SYSTEM_PROMPT
    assert "get_references" in CHAT_SYSTEM_PROMPT
    assert "/cites" in CHAT_SYSTEM_PROMPT
    assert "/refs" in CHAT_SYSTEM_PROMPT
