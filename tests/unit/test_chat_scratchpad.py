"""Tests for the cross-turn scratchpad cache (T2.3)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from research_agent.chat import run_chat
from research_agent.chat.router import (
    SCRATCHPAD_MAX_CHARS,
    SCRATCHPAD_TOOLS,
    _build_messages,
    _refresh_scratchpad,
)
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall


def _cfg(d: Path) -> Config:
    return Config(data_dir=d, api_key="sk-x")


def _make_session(d: Path) -> ChatSession:
    return ChatSession.create(
        cfg=_cfg(d),
        llm=MockLLMProvider([]),
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )


def test_refresh_scratchpad_creates_entry(config_dir: Path) -> None:
    session = _make_session(config_dir)
    _refresh_scratchpad(session, "list_ideas", "idea A\nidea B")
    entries = [
        m for m in session.memory.messages
        if m.metadata.get("kind") == "scratchpad"
    ]
    assert len(entries) == 1
    assert entries[0].metadata.get("tool_name") == "list_ideas"
    assert "idea A" in entries[0].content


def test_refresh_scratchpad_replaces_prior_entry(config_dir: Path) -> None:
    session = _make_session(config_dir)
    _refresh_scratchpad(session, "list_ideas", "stale snapshot")
    _refresh_scratchpad(session, "list_ideas", "fresh snapshot")
    live = [
        m for m in session.memory.messages
        if m.metadata.get("kind") == "scratchpad"
    ]
    assert len(live) == 1
    assert "fresh snapshot" in live[0].content
    # The stale one is now demoted to tool_log (kept for audit).
    demoted = [
        m for m in session.memory.messages
        if m.metadata.get("kind") == "tool_log"
        and "stale snapshot" in m.content
    ]
    assert demoted, "stale entry should be demoted to tool_log, not deleted"


def test_refresh_scratchpad_caps_long_content(config_dir: Path) -> None:
    session = _make_session(config_dir)
    huge = "Q" * (SCRATCHPAD_MAX_CHARS * 3)
    _refresh_scratchpad(session, "search_arxiv", huge)
    entry = next(
        m for m in session.memory.messages
        if m.metadata.get("kind") == "scratchpad"
    )
    assert len(entry.content) < SCRATCHPAD_MAX_CHARS + 100


def test_build_messages_hoists_scratchpad(config_dir: Path) -> None:
    session = _make_session(config_dir)
    _refresh_scratchpad(session, "queue_list", "Pending: 1706.03762")
    session.memory.append("user", "what's queued?")
    msgs = _build_messages(session)
    contents = [m.content for m in msgs]
    assert any("1706.03762" in c for c in contents)
    # Scratchpad must come before the user turn so the model sees it
    # as context, not as a reply.
    pad_idx = next(i for i, c in enumerate(contents) if "1706.03762" in c)
    user_idx = next(i for i, c in enumerate(contents) if "what's queued?" in c)
    assert pad_idx < user_idx


def test_scratchpad_tool_set_is_idempotent_lightweight_only() -> None:
    """Guard against accidentally adding state-mutating or heavy tools
    to the scratchpad set (which would re-pin stale state next turn)."""
    forbidden = {
        "queue_add", "discuss_idea", "load_paper",
        "draft_section", "draft_figure", "save_draft_to_file",
        "set_config", "train_style", "build_fingerprint",
        "update_fingerprint", "ingest_local_papers",
        "save_current_idea", "revise_draft", "check_self_plagiarism",
    }
    assert not (SCRATCHPAD_TOOLS & forbidden), (
        "Only idempotent + read-only tools belong in SCRATCHPAD_TOOLS"
    )


def test_tool_loop_promotes_search_result_to_scratchpad(
    config_dir: Path,
) -> None:
    """End-to-end: agent calls search_arxiv, the (capped) result lands
    in WorkingMemory tagged scratchpad so the next turn picks it up."""
    import research_agent.chat.tools as chat_tools

    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(
        [
            # Turn 1: model calls search_arxiv, then replies with text.
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_arxiv",
                        arguments=json.dumps({"query": "transformer"}),
                    ),
                ),
            ),
            "Here are the hits.",
            # Turn 2: model just replies. The scratchpad from turn 1
            # should be visible in this turn's prompt.
            "Got it, here's a follow-up.",
        ]
    )

    original = chat_tools.LLM_TOOLS["search_arxiv"]
    canned = (
        "Top arxiv hits for 'transformer':\n"
        "- 1706.03762 (2017) Attention Is All You Need"
    )

    def stub(session: ChatSession, args: dict[str, object]) -> str:
        return canned

    chat_tools.LLM_TOOLS["search_arxiv"] = original.__class__(
        name=original.name, schema=original.schema, executor=stub
    )
    try:
        # Two real turns + /exit. The scratchpad written during turn 1
        # must show up in the prompt for turn 2.
        inputs = iter(
            [
                "Find me a transformer paper.",
                "anything else I should know?",
                "/exit",
            ]
        )
        run_chat(
            cfg,
            llm,
            console,
            input_fn=lambda _: next(inputs),
            use_chroma=False,
        )
    finally:
        chat_tools.LLM_TOOLS["search_arxiv"] = original

    # The third LLM call (turn 2's only round) must have included the
    # scratchpad hoisted as a system message.
    assert len(llm.calls) >= 3, "expected at least 3 LLM round-trips"
    turn2_messages = llm.calls[-1]
    found = any(
        "[scratchpad search_arxiv]" in m.content and "1706.03762" in m.content
        for m in turn2_messages
    )
    assert found, (
        "scratchpad entry was not visible to subsequent LLM round"
    )
