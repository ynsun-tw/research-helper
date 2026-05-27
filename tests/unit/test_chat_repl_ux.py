"""Tests for the REPL UX layer: streaming, ``?`` alias, Ctrl+C handling,
state-aware prompt, slash completer.

These exercise the boundaries we introduced when we replaced
``console.input`` with prompt_toolkit and the non-streaming LLM call with
``chat_with_tools_stream``. They deliberately avoid spinning up a real
``PromptSession`` (no tty in CI) and instead pass an ``input_fn`` lambda
so the router stays on its fallback path. The PromptSession factory is
covered by an isolated import + construction test.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from research_agent.chat import run_chat
from research_agent.chat.prompt_ui import (
    build_prompt_session,
    render_state_prompt,
)
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import (
    ChatMessage,
    ChatResponse,
    MockLLMProvider,
    ToolCall,
)
from research_agent.core.paper import Paper, Section


def _cfg(tmp_dir: Path) -> Config:
    return Config(data_dir=tmp_dir, api_key="sk-x")


# ---------------------------------------------------------------------------
# LLM streaming roundtrip
# ---------------------------------------------------------------------------


def test_mock_provider_streams_full_content_via_callback() -> None:
    """``chat_with_tools_stream`` on the mock provider should feed the full
    queued content through ``on_content_delta`` exactly once and return the
    same ``ChatResponse`` it would have returned non-streaming."""
    llm = MockLLMProvider(["Hello, world."])
    chunks: list[str] = []
    resp = llm.chat_with_tools_stream(
        [ChatMessage(role="user", content="hi")],
        on_content_delta=chunks.append,
    )
    assert resp.content == "Hello, world."
    assert chunks == ["Hello, world."]
    assert resp.tool_calls == ()


def test_mock_provider_stream_with_tool_call_skips_callback() -> None:
    """Tool-call-only responses (no text) must not invoke the delta callback
    — otherwise we'd render an empty line during agent loops."""
    llm = MockLLMProvider(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="noop", arguments="{}"),),
            )
        ]
    )
    chunks: list[str] = []
    resp = llm.chat_with_tools_stream(
        [ChatMessage(role="user", content="hi")],
        on_content_delta=chunks.append,
    )
    assert chunks == []
    assert resp.tool_calls and resp.tool_calls[0].name == "noop"


def test_router_streams_assistant_text_into_console(config_dir: Path) -> None:
    """End-to-end: a plain-text user turn drives ``chat_with_tools_stream``
    and the streamed delta ends up in the console buffer."""
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120, force_terminal=False)
    llm = MockLLMProvider(["Try /search transformer first."])
    inputs = iter(["What should I read?", "/exit"])

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )

    out = console.file.getvalue()
    assert "Try /search transformer" in out


# ---------------------------------------------------------------------------
# ``?`` / ``/?`` help alias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["?", "/?", "help"])
def test_help_alias_dispatches_to_help_command(
    config_dir: Path, alias: str
) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([])
    inputs = iter([alias, "/exit"])

    code = run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )

    assert code == 0
    out = console.file.getvalue()
    assert "/help" in out
    assert "/search" in out
    # Alias must NOT have leaked into the LLM call queue.
    assert not llm.calls


# ---------------------------------------------------------------------------
# Ctrl+C smart handling
# ---------------------------------------------------------------------------


def test_single_ctrl_c_warns_then_second_exits(config_dir: Path) -> None:
    """First KeyboardInterrupt should arm a warning. Second one (with no
    successful input in between) actually exits the loop."""
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([])

    interrupts = iter([KeyboardInterrupt(), KeyboardInterrupt()])

    def fake_input(_prompt: str) -> str:
        raise next(interrupts)

    code = run_chat(cfg, llm, console, input_fn=fake_input, use_chroma=False)

    out = console.file.getvalue()
    assert code == 0
    assert "Press Ctrl+C again" in out
    assert "Interrupted" in out


def test_ctrl_c_then_successful_input_resets_arm(
    config_dir: Path,
) -> None:
    """After a successful turn between two Ctrl+Cs, the second Ctrl+C should
    re-arm (warn) rather than exit. This keeps "I was just clearing the
    buffer" interactions safe."""
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(["Sure."])

    events: list[object] = [
        KeyboardInterrupt(),  # arm warning
        "What's new?",  # successful input; arm should reset
        KeyboardInterrupt(),  # should warn again, not exit
        "/exit",
    ]
    it = iter(events)

    def fake_input(_prompt: str) -> str:
        nxt = next(it)
        if isinstance(nxt, BaseException):
            raise nxt
        return str(nxt)

    run_chat(cfg, llm, console, input_fn=fake_input, use_chroma=False)
    out = console.file.getvalue()
    # Two distinct "press again" warnings imply the arm was reset after
    # the successful turn.
    assert out.count("Press Ctrl+C again") == 2


# ---------------------------------------------------------------------------
# State-aware prompt rendering
# ---------------------------------------------------------------------------


def _make_session(cfg: Config) -> ChatSession:
    return ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )


def test_render_state_prompt_no_anchor_no_idea(config_dir: Path) -> None:
    session = _make_session(_cfg(config_dir))
    text = "".join(frag for _style, frag in render_state_prompt(session))
    assert text == "You> "


def test_render_state_prompt_with_anchor(config_dir: Path) -> None:
    session = _make_session(_cfg(config_dir))
    session.anchor_paper = Paper(
        id="arxiv:1706.03762",
        title="Attention",
        abstract="",
        sections=[Section("Intro", "x")],
        full_text="x",
    )
    text = "".join(frag for _style, frag in render_state_prompt(session))
    assert "arxiv:1706.03762" in text
    assert text.endswith("> ")


def test_render_state_prompt_with_anchor_and_idea(config_dir: Path) -> None:
    session = _make_session(_cfg(config_dir))
    session.anchor_paper = Paper(
        id="arxiv:1706.03762",
        title="Attention",
        abstract="",
        sections=[Section("Intro", "x")],
        full_text="x",
    )
    session.current_idea_id = "abcdef1234567890"
    text = "".join(frag for _style, frag in render_state_prompt(session))
    assert "arxiv:1706.03762" in text
    # Idea id is truncated to 8 chars.
    assert "idea:abcdef12" in text
    assert "idea:abcdef1234" not in text


# ---------------------------------------------------------------------------
# PromptSession factory + completer
# ---------------------------------------------------------------------------


def test_build_prompt_session_writes_history_dir(config_dir: Path) -> None:
    cfg = _cfg(config_dir)
    ps = build_prompt_session(cfg, ["help", "exit", "search"])
    # File doesn't have to exist yet, but the directory must.
    assert (Path(cfg.data_dir) / "repl_history").parent.exists()
    # Completer should be wired up.
    assert ps.completer is not None


def test_slash_completer_matches_prefix(config_dir: Path) -> None:
    from prompt_toolkit.document import Document

    cfg = _cfg(config_dir)
    ps = build_prompt_session(cfg, ["help", "exit", "search", "history"])
    completer = ps.completer
    assert completer is not None

    doc = Document(text="/he", cursor_position=3)
    completions = list(completer.get_completions(doc, complete_event=None))  # type: ignore[arg-type]
    names = sorted(c.text for c in completions)
    assert names == ["help"]

    # After a space, we're in args territory — no command suggestions.
    doc2 = Document(text="/search ", cursor_position=8)
    assert list(completer.get_completions(doc2, complete_event=None)) == []  # type: ignore[arg-type]

    # Without leading slash, no suggestions (otherwise plain text would
    # surface noisy popups while the user is typing an LLM prompt).
    doc3 = Document(text="he", cursor_position=2)
    assert list(completer.get_completions(doc3, complete_event=None)) == []  # type: ignore[arg-type]
