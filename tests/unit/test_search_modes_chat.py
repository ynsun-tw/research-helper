"""Chat-side wiring of /search --mode and search_arxiv(mode=...) (T3.1.3.3)."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    LLM_TOOLS,
    SLASH_COMMANDS,
    _extract_mode_flag,
    cmd_search,
    exec_search_arxiv,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.search.arxiv_search import ArxivSearchHit


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


# ------------------------- argument parsing --------------------------


def test_extract_mode_flag_with_space_form() -> None:
    mode, rest = _extract_mode_flag("--mode theoretical attention transformer")
    assert mode == "theoretical"
    assert rest == "attention transformer"


def test_extract_mode_flag_with_equals_form() -> None:
    mode, rest = _extract_mode_flag("--mode=applied attention transformer")
    assert mode == "applied"
    assert rest == "attention transformer"


def test_extract_mode_flag_anywhere_in_args() -> None:
    """--mode shouldn't have to be the first token."""
    mode, rest = _extract_mode_flag("attention --mode theoretical transformer")
    assert mode == "theoretical"
    assert rest == "attention transformer"


def test_extract_mode_flag_absent_returns_none() -> None:
    mode, rest = _extract_mode_flag("attention transformer")
    assert mode is None
    assert rest == "attention transformer"


def test_extract_mode_flag_handles_group_value() -> None:
    mode, rest = _extract_mode_flag("--mode=group:Karpathy transformer")
    assert mode == "group:Karpathy"
    assert rest == "transformer"


# --------------------------- /search slash --------------------------


def test_cmd_search_passes_mode_to_resolver(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    captured: dict[str, object] = {}

    def fake_search(query, *, max_results=5, mode=None, **_kw):  # type: ignore[no-untyped-def]
        captured["query"] = query
        captured["mode"] = mode
        return [ArxivSearchHit("1706.03762", "Attention Is All You Need", "")]

    monkeypatch.setattr("research_agent.chat.tools.search_arxiv_papers", fake_search)
    # Skip the LLM-based relevance scorer to keep the test focused.
    monkeypatch.setattr(
        "research_agent.chat.tools._score_hits",
        lambda session, query, hits: hits,
    )
    try:
        cmd_search(session, "--mode theoretical attention transformer")
        assert captured["mode"] == "theoretical"
        assert captured["query"] == "attention transformer"
        out = console.file.getvalue()
        assert "theoretical" in out  # hint surfaced to the user
    finally:
        session.close()


def test_cmd_search_warns_on_unknown_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)

    def fake_search(query, *, max_results=5, mode=None, **_kw):  # type: ignore[no-untyped-def]
        return [ArxivSearchHit("1706.03762", "x", "")]

    monkeypatch.setattr("research_agent.chat.tools.search_arxiv_papers", fake_search)
    monkeypatch.setattr(
        "research_agent.chat.tools._score_hits",
        lambda session, query, hits: hits,
    )
    try:
        cmd_search(session, "--mode bogus attention")
        out = console.file.getvalue()
        assert "Unknown" in out
    finally:
        session.close()


def test_cmd_search_with_group_mode_quoted_multiword_author(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-word author names must be quoted - shlex respects that."""
    session, console = _make_session(tmp_path)
    monkeypatch.setattr(
        "research_agent.chat.tools.search_arxiv_papers",
        lambda query, *, max_results=5, mode=None, **_kw: [
            ArxivSearchHit("1.1", "x", "")
        ],
    )
    monkeypatch.setattr(
        "research_agent.chat.tools._score_hits",
        lambda session, query, hits: hits,
    )
    try:
        cmd_search(session, '--mode "group:Andrej Karpathy" attention')
        out = console.file.getvalue()
        assert "Andrej Karpathy" in out
    finally:
        session.close()


def test_extract_mode_flag_with_quoted_multiword_value() -> None:
    mode, rest = _extract_mode_flag('--mode "group:Andrej Karpathy" attention')
    assert mode == "group:Andrej Karpathy"
    assert rest == "attention"


def test_extract_mode_flag_handles_unbalanced_quotes() -> None:
    """Bad quoting shouldn't crash the search - fall back to plain split."""
    mode, rest = _extract_mode_flag('--mode theoretical attention "transformer')
    # shlex would raise; fallback splits on whitespace.
    assert mode == "theoretical"
    assert "transformer" in rest


# --------------------------- search_arxiv tool --------------------------


def test_search_arxiv_tool_schema_advertises_mode() -> None:
    schema = LLM_TOOLS["search_arxiv"].schema["function"]
    props = schema["parameters"]["properties"]
    assert "mode" in props
    assert "theoretical" in props["mode"]["description"]


def test_exec_search_arxiv_passes_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _ = _make_session(tmp_path)
    captured: dict[str, object] = {}

    def fake_search(query, *, max_results=5, mode=None, **_kw):  # type: ignore[no-untyped-def]
        captured["query"] = query
        captured["mode"] = mode
        return [ArxivSearchHit("1706.03762", "x", "")]

    monkeypatch.setattr("research_agent.chat.tools.search_arxiv_papers", fake_search)
    monkeypatch.setattr(
        "research_agent.chat.tools._score_hits",
        lambda session, query, hits: hits,
    )
    try:
        out = exec_search_arxiv(
            session,
            {"query": "attention transformer", "mode": "applied"},
        )
        assert captured["mode"] == "applied"
        assert "Searched arXiv" in out or "1706.03762" in out
    finally:
        session.close()


def test_exec_search_arxiv_rejects_unknown_mode(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    try:
        out = exec_search_arxiv(
            session, {"query": "attention", "mode": "recreational"}
        )
        assert out.startswith("Error:")
        assert "Unknown" in out
    finally:
        session.close()


def test_slash_search_usage_advertises_mode_flag() -> None:
    usage = SLASH_COMMANDS["search"].usage
    assert "--mode" in usage
