"""Tests for intra-loop tool-call deduplication (T3.2)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from research_agent.chat import run_chat
from research_agent.chat.router import _tool_call_cache_key
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall


def _cfg(d: Path) -> Config:
    return Config(data_dir=d, api_key="sk-x")


def test_cache_key_canonicalises_json_arguments() -> None:
    tc1 = ToolCall(id="1", name="search_arxiv", arguments='{"q":"a","n":3}')
    tc2 = ToolCall(id="2", name="search_arxiv", arguments='{"n":3,"q":"a"}')
    assert _tool_call_cache_key(tc1) == _tool_call_cache_key(tc2)


def test_cache_key_differs_on_different_args() -> None:
    tc1 = ToolCall(id="1", name="search_arxiv", arguments='{"q":"a"}')
    tc2 = ToolCall(id="2", name="search_arxiv", arguments='{"q":"b"}')
    assert _tool_call_cache_key(tc1) != _tool_call_cache_key(tc2)


def test_cache_key_handles_invalid_json_safely() -> None:
    tc = ToolCall(id="x", name="search_arxiv", arguments="not json {")
    name, canon = _tool_call_cache_key(tc)
    assert name == "search_arxiv"
    assert canon == "not json {"


def test_duplicate_tool_call_in_same_turn_is_short_circuited(
    config_dir: Path,
) -> None:
    """If the LLM emits the same tool call twice in one turn, the second
    invocation should hit the cache and never run the executor again."""
    import research_agent.chat.tools as chat_tools

    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    # Two rounds: each round emits a single tool_call to the same
    # function with the same args.
    args = json.dumps({"query": "transformer", "max_results": 3})
    llm = MockLLMProvider(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="search_arxiv", arguments=args),),
            ),
            ChatResponse(
                content="",
                tool_calls=(ToolCall(id="c2", name="search_arxiv", arguments=args),),
            ),
            "Done.",
        ]
    )

    call_count = 0

    def counting_stub(session: ChatSession, args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        return f"hit #{call_count}: 1706.03762"

    original = chat_tools.LLM_TOOLS["search_arxiv"]
    chat_tools.LLM_TOOLS["search_arxiv"] = original.__class__(
        name=original.name, schema=original.schema, executor=counting_stub
    )
    try:
        inputs = iter(["find me a transformer paper", "/exit"])
        run_chat(
            cfg,
            llm,
            console,
            input_fn=lambda _: next(inputs),
            use_chroma=False,
        )
    finally:
        chat_tools.LLM_TOOLS["search_arxiv"] = original

    # Executor must have run exactly once even though the model called
    # it twice — the second tool message comes from the dedup cache.
    assert call_count == 1, (
        f"expected exactly one executor invocation; got {call_count}"
    )
    # The second tool result should carry the [deduplicated] marker.
    last_call = llm.calls[-1]
    tool_messages = [m for m in last_call if m.role == "tool"]
    assert any("deduplicated" in m.content for m in tool_messages), (
        "dedup marker not surfaced to the LLM"
    )
