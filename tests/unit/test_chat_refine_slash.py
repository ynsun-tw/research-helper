"""`/refine` slash + LLM tool (M3 T3.2.3.3).

Drives the chat-side interaction: build context, call
``Searcher.suggest_refinement``, render banner, prompt the user
(`input_fn`) for accept / edit / skip, and dispatch into ``cmd_search``
on accept. We stub the Searcher so we never hit the LLM and never hit
the network (cmd_search is also stubbed to record args).
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.agents.searcher import SearchSuggestion
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    _format_search_args,
    cmd_refine,
    exec_suggest_search_refinement,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


@pytest.fixture
def make_session(tmp_path):
    inputs: list[str] = []

    def _factory(prompt_inputs: list[str] | None = None):
        nonlocal inputs
        inputs = list(prompt_inputs or [])
        cfg = Config(data_dir=tmp_path, api_key="sk-x")
        console = Console(file=StringIO(), width=140)

        def _input(_: str) -> str:
            return inputs.pop(0) if inputs else ""

        return ChatSession.create(
            cfg=cfg,
            llm=MockLLMProvider([]),
            console=console,
            input_fn=_input,
            use_chroma=False,
        )

    return _factory


def _seed_context(session: ChatSession) -> None:
    session.memory.append("user", "I'm exploring efficient attention.")
    session.memory.append("analyst", "Paper proposes block-sparse pattern.")
    session.memory.append("critic", "But the bottleneck is memory bandwidth.")


def _stub_suggestion(
    session: ChatSession, suggestion: SearchSuggestion
) -> None:
    session.searcher.suggest_refinement = lambda *a, **kw: suggestion  # type: ignore[method-assign]


def _stub_cmd_search(session: ChatSession, sink: list[str]) -> None:
    """Replace cmd_search globally for one test so /refine's dispatch is
    observable without hitting the network."""
    import research_agent.chat.tools as tools_module

    def _fake(s: ChatSession, args: str) -> None:
        sink.append(args)

    tools_module.cmd_search = _fake  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _restore_cmd_search():
    """Snapshot the real cmd_search and restore it after each test."""
    import research_agent.chat.tools as tools_module

    original = tools_module.cmd_search
    yield
    tools_module.cmd_search = original  # type: ignore[assignment]


# ---------------------------------------------------------------- /refine


def test_refine_silent_without_context(make_session) -> None:
    session = make_session()
    cmd_refine(session, "")
    out = session.console.file.getvalue()
    assert "Not enough conversation yet" in out


def test_refine_silent_when_searcher_returns_empty_query(make_session) -> None:
    session = make_session()
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="", mode=None, reason="thin", confidence=0.0),
    )
    cmd_refine(session, "")
    out = session.console.file.getvalue()
    assert "Searcher returned no refined query" in out


def test_refine_accepts_and_dispatches_cmd_search(make_session) -> None:
    session = make_session(["y"])  # user types y
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(
            query="FlashAttention KV cache", mode="applied",
            reason="bandwidth bottleneck", confidence=0.8,
        ),
    )
    sink: list[str] = []
    _stub_cmd_search(session, sink)
    cmd_refine(session, "")
    assert sink == ["--mode applied FlashAttention KV cache"]
    out = session.console.file.getvalue()
    assert "FlashAttention KV cache" in " ".join(out.split())


def test_refine_skip_does_not_dispatch(make_session) -> None:
    session = make_session(["s"])
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="x", mode=None, reason="r", confidence=0.5),
    )
    sink: list[str] = []
    _stub_cmd_search(session, sink)
    cmd_refine(session, "")
    assert sink == []
    assert "Skipped" in session.console.file.getvalue()


def test_refine_edit_uses_user_query(make_session) -> None:
    session = make_session(["e", "my custom query"])
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="suggested", mode="theoretical", reason="r", confidence=0.6),
    )
    sink: list[str] = []
    _stub_cmd_search(session, sink)
    cmd_refine(session, "")
    assert sink == ["--mode theoretical my custom query"]


def test_refine_edit_empty_input_aborts(make_session) -> None:
    session = make_session(["e", ""])
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="suggested", mode=None, reason="r", confidence=0.6),
    )
    sink: list[str] = []
    _stub_cmd_search(session, sink)
    cmd_refine(session, "")
    assert sink == []
    assert "No edit; skipping" in session.console.file.getvalue()


def test_refine_eof_treated_as_skip(make_session) -> None:
    session = make_session()  # no inputs queued; input_fn returns ""
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="x", mode=None, reason="r", confidence=0.5),
    )
    sink: list[str] = []
    _stub_cmd_search(session, sink)
    cmd_refine(session, "")
    assert sink == []


def test_refine_writes_system_memory(make_session) -> None:
    session = make_session(["s"])
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="zebra", mode=None, reason="r", confidence=0.5),
    )
    cmd_refine(session, "")
    sys_msgs = [m for m in session.memory.messages if m.role == "system"]
    assert any("Refinement suggestion" in m.content for m in sys_msgs)


# ---------------------------------------------------------------- LLM tool


def test_exec_suggest_search_refinement_no_context(make_session) -> None:
    session = make_session()
    out = exec_suggest_search_refinement(session, {})
    assert "Error" in out
    assert "discuss" in out.lower() or "load" in out.lower()


def test_exec_suggest_search_refinement_returns_payload(make_session) -> None:
    session = make_session()
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(
            query="quantized inference",
            mode="applied",
            reason="critic wants benchmarks",
            confidence=0.7,
        ),
    )
    out = exec_suggest_search_refinement(session, {})
    assert "quantized inference" in out
    assert "applied" in out
    assert "search_arxiv" in out


def test_exec_suggest_search_refinement_empty_suggestion(make_session) -> None:
    session = make_session()
    _seed_context(session)
    _stub_suggestion(
        session,
        SearchSuggestion(query="", mode=None, reason="", confidence=0.0),
    )
    out = exec_suggest_search_refinement(session, {})
    assert "no refinement" in out.lower()


# ---------------------------------------------------------------- helpers


def test_format_search_args_no_mode() -> None:
    assert _format_search_args("foo bar", None) == "foo bar"


def test_format_search_args_simple_mode() -> None:
    assert _format_search_args("foo", "applied") == "--mode applied foo"


def test_format_search_args_quotes_group_with_spaces() -> None:
    assert (
        _format_search_args("transformers", "group:Andrej Karpathy")
        == '--mode "group:Andrej Karpathy" transformers'
    )
