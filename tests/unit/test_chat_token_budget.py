"""Tests for token-budget controls in the chat loop.

Covers the three optimizations that touch runtime behavior (not just
prompt text):

* tool results that exceed ``MAX_TOOL_RESULT_CHARS`` are truncated with
  an explicit ``elided`` marker before they reach the LLM
* tool-log messages persisted into ``WorkingMemory`` are excluded from
  the prompt assembled by ``_build_messages`` (so they don't get
  double-charged the next turn)
* every assistant turn prints a token-usage footer

Steps that only touch the system prompt / tool-schema text are
exercised by the existing chat tests — those still pass after the
slim-down, which is the integration-level confirmation we want.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from research_agent.chat import run_chat
from research_agent.chat.router import (
    MAX_HISTORY_TOKENS,
    MAX_TOOL_RESULT_CHARS,
    _build_messages,
    _cap_tool_result,
    _select_history_within_budget,
)
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall
from research_agent.memory.working_memory import MemoryMessage


def _cfg(tmp_dir: Path) -> Config:
    return Config(data_dir=tmp_dir, api_key="sk-x")


def test_cap_tool_result_truncates_long_payload() -> None:
    payload = "x" * (MAX_TOOL_RESULT_CHARS + 5000)
    capped = _cap_tool_result(payload)
    assert len(capped) < len(payload)
    assert capped.startswith("x" * 100)
    assert "elided" in capped
    assert "narrower" in capped


def test_cap_tool_result_passes_short_payloads_through() -> None:
    payload = "short result"
    assert _cap_tool_result(payload) is payload


def test_build_messages_skips_tool_log_entries(config_dir: Path) -> None:
    cfg = _cfg(config_dir)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )
    session.memory.append("user", "what should I read?")
    session.memory.append("assistant", "Let me check.")
    session.memory.append(
        "system",
        "[tool search_arxiv] 17 hits",
        metadata={"kind": "tool_log"},
    )
    session.memory.append("user", "now read the second one")

    msgs = _build_messages(session)
    contents = [m.content for m in msgs]
    assert any("what should I read?" in c for c in contents)
    assert any("now read the second one" in c for c in contents)
    assert not any("[tool search_arxiv]" in c for c in contents)


def test_chat_turn_prints_token_footer(config_dir: Path) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(["Sure, let me think about that."])
    inputs = iter(["Plain question.", "/exit"])

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    out = console.file.getvalue()
    # Footer pattern: [~N in → ~M out tokens · K round]
    assert " in →" in out
    assert "tokens" in out


def test_select_history_keeps_recent_turns_within_budget() -> None:
    messages = [
        MemoryMessage(role="user", content="early " * 50),
        MemoryMessage(role="assistant", content="reply 1"),
        MemoryMessage(role="user", content="reply 2"),
        MemoryMessage(role="assistant", content="reply 3"),
    ]
    picked = _select_history_within_budget(messages, max_tokens=10_000)
    assert picked == messages  # whole list fits


def test_select_history_drops_oldest_when_over_budget() -> None:
    # The middle message alone is ~5000 tokens; combined with anything
    # it busts the 4000-tok budget so "oldest" must get dropped.
    messages = [
        MemoryMessage(role="user", content="oldest"),
        MemoryMessage(role="assistant", content="x" * 20_000),
        MemoryMessage(role="user", content="recent question"),
    ]
    picked = _select_history_within_budget(
        messages, max_tokens=4000, min_turns=1
    )
    contents = [m.content for m in picked]
    assert "recent question" in contents
    assert "oldest" not in contents


def test_select_history_respects_min_turns_floor() -> None:
    """Even when the recent message is oversized, the floor of
    min_turns must hold so the model still sees the current turn."""
    huge = "x" * 100_000
    messages = [
        MemoryMessage(role="user", content="filler"),
        MemoryMessage(role="assistant", content="ok"),
        MemoryMessage(role="user", content=huge),  # massive recent msg
    ]
    picked = _select_history_within_budget(
        messages, max_tokens=1000, min_turns=2
    )
    assert len(picked) >= 2
    assert picked[-1].content == huge


def test_build_messages_token_budget_starves_no_recent_turns(
    config_dir: Path,
) -> None:
    """A fat /insights-style system message must not push the live user
    turn out of the prompt."""
    cfg = _cfg(config_dir)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )
    fat = "INSIGHT " * 4000  # ~32 KB, well over the default 4K-token budget
    session.memory.append("system", fat)
    session.memory.append("user", "earlier question")
    session.memory.append("assistant", "earlier reply")
    session.memory.append("user", "what about now?")

    msgs = _build_messages(session, max_tokens=MAX_HISTORY_TOKENS)
    contents = [m.content for m in msgs]
    assert any("what about now?" in c for c in contents)
    # The fat insights message must have been dropped to keep the
    # recent turns inside the budget.
    assert not any(c == fat for c in contents)


def test_slash_command_appends_are_tagged_tool_log(
    config_dir: Path,
) -> None:
    """Slash commands that render large summaries to the user must tag
    their memory appends as tool_log so the next LLM turn doesn't
    re-pay for that text."""
    cfg = _cfg(config_dir)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )
    # Simulate what /insights, /search, /cites, /refs would persist
    # (all use the same metadata={"kind": "tool_log"} contract now).
    session.memory.append(
        "system",
        "Insights report (period=None):\n# Huge markdown ...",
        metadata={"kind": "tool_log"},
    )
    session.memory.append("user", "what should I work on next?")
    msgs = _build_messages(session)
    contents = [m.content for m in msgs]
    assert any("what should I work on next?" in c for c in contents)
    assert not any("Insights report" in c for c in contents)


def test_cap_tool_result_uses_per_tool_override() -> None:
    """A tool with a larger ``result_cap_chars`` should NOT be trimmed
    even when the payload exceeds the global cap (T3.3)."""
    from research_agent.chat.tools import LLM_TOOLS

    # draft_section is configured with a 50K cap in _RESULT_CAP_OVERRIDES.
    assert LLM_TOOLS["draft_section"].result_cap_chars == 50_000
    payload = "x" * (MAX_TOOL_RESULT_CHARS + 4000)
    capped = _cap_tool_result(payload, tool_name="draft_section")
    # Whole payload survives because it's below 50K.
    assert capped == payload


def test_cap_tool_result_uses_tighter_per_tool_override() -> None:
    """list_ideas has a 2K cap — its output should be trimmed harder."""
    from research_agent.chat.tools import LLM_TOOLS

    assert LLM_TOOLS["list_ideas"].result_cap_chars == 2_000
    payload = "x" * 5_000
    capped = _cap_tool_result(payload, tool_name="list_ideas")
    assert "elided" in capped
    assert capped.startswith("x" * 100)
    assert len(capped) < 2_300  # 2000 + elision marker overhead


def test_cap_tool_result_falls_back_to_global_for_unknown_tool() -> None:
    payload = "x" * (MAX_TOOL_RESULT_CHARS + 1000)
    capped = _cap_tool_result(payload, tool_name="no-such-tool")
    assert "elided" in capped


def test_tool_loop_caps_long_results_and_tags_memory(
    config_dir: Path,
) -> None:
    """A tool that returns a giant payload should land in WorkingMemory
    flagged as ``tool_log`` AND the LLM should never see the un-capped
    string in ``messages``."""
    import research_agent.chat.tools as chat_tools

    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    # Two-step canned dialogue: first response triggers our tool, second
    # response is the final assistant text after the tool returns.
    huge = "Z" * (MAX_TOOL_RESULT_CHARS + 4000)
    llm = MockLLMProvider(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="research_insights",
                        arguments=json.dumps({}),
                    ),
                ),
            ),
            "Done. Here's the summary.",
        ]
    )

    # Stub research_insights to return the huge payload deterministically.
    original = chat_tools.LLM_TOOLS["research_insights"]

    def big_executor(session: ChatSession, args: dict[str, object]) -> str:
        return huge

    chat_tools.LLM_TOOLS["research_insights"] = original.__class__(
        name=original.name, schema=original.schema, executor=big_executor
    )
    try:
        inputs = iter(["Show me the rollup.", "/exit"])
        run_chat(
            cfg,
            llm,
            console,
            input_fn=lambda _prompt: next(inputs),
            use_chroma=False,
        )
    finally:
        chat_tools.LLM_TOOLS["research_insights"] = original

    # The huge payload should never have been sent to the mock LLM
    # un-capped. ``llm.calls`` holds every messages list the agent sent.
    for call in llm.calls:
        for msg in call:
            assert len(msg.content) <= MAX_TOOL_RESULT_CHARS + 500, (
                "tool result reached LLM without being capped"
            )
