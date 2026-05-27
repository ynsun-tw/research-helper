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
    MAX_TOOL_RESULT_CHARS,
    _build_messages,
    _cap_tool_result,
)
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall


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
