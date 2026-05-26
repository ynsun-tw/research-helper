"""Tests for the LLM agent loop in chat/router.py (Phase 2)."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

import research_agent.chat.tools as chat_tools
from research_agent.chat import run_chat
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall
from research_agent.core.paper import Paper, Section

ANALYST_JSON = json.dumps(
    {
        "contributions": ["C1"],
        "method_insights": ["M1"],
        "potential_impact": "Impact",
        "related_work": ["R1"],
        "claimed_vs_evidence": [],
        "confidence": 0.8,
    }
)
CRITIC_JSON = json.dumps(
    {
        "objections": ["O1"],
        "support_score": 7,
        "score_reason": "Good",
        "honesty_note": "",
    }
)
IDEA_ANALYST_JSON = json.dumps(
    {
        "supports": ["Solid premise"],
        "suggestions": ["More baselines"],
        "evidence": [],
        "confidence": 0.8,
    }
)
IDEA_CRITIC_JSON = json.dumps(
    {
        "objections": ["Risky assumption"],
        "support_score": 7,
        "score_reason": "Viable with fixes",
        "suggestions": [],
        "honesty_note": "",
    }
)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer architecture.",
        sections=[Section("Intro", "Self-attention.")],
        full_text="Transformer details.",
    )


def _cfg(tmp_dir) -> Config:
    return Config(data_dir=tmp_dir, api_key="sk-x")


def _tool_call_response(name: str, arguments: dict) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=(
            ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(arguments)),
        ),
    )


def test_llm_can_call_load_paper_then_summarize(
    config_dir,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    monkeypatch.setattr(
        chat_tools,
        "_load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    llm = MockLLMProvider(
        [
            # Turn 1: LLM asks to load_paper
            _tool_call_response("load_paper", {"source": "attention"}),
            ANALYST_JSON,
            CRITIC_JSON,
            # Turn 2: LLM produces final text summary
            ChatResponse(content="Loaded the Transformer paper for you."),
        ]
    )

    inputs = iter(["please load the attention paper", "/exit"])
    code = run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    assert code == 0
    out = console.file.getvalue()
    assert "load_paper" in out
    assert "Loaded the Transformer paper" in out


def test_llm_can_chain_load_then_discuss(
    config_dir,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    monkeypatch.setattr(
        chat_tools,
        "_load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    llm = MockLLMProvider(
        [
            _tool_call_response("load_paper", {"source": "attention"}),
            ANALYST_JSON,
            CRITIC_JSON,
            _tool_call_response(
                "discuss_idea", {"idea": "Apply attention to molecules"}
            ),
            IDEA_ANALYST_JSON,
            IDEA_CRITIC_JSON,
            ChatResponse(content="Debate score: 7/9 — proceed with caveats."),
        ]
    )

    inputs = iter(
        ["Load Attention then debate applying it to biology", "/exit"]
    )
    code = run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    assert code == 0
    out = console.file.getvalue()
    assert "load_paper" in out
    assert "discuss_idea" in out
    assert "Debate score: 7/9" in out


def test_unknown_tool_returns_error_to_llm(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    llm = MockLLMProvider(
        [
            _tool_call_response("nonexistent_tool", {"x": 1}),
            ChatResponse(content="Sorry, that tool is not available."),
        ]
    )

    inputs = iter(["do something weird", "/exit"])
    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    out = console.file.getvalue()
    assert "unknown tool" in out.lower() or "Model requested unknown tool" in out
    assert "Sorry, that tool is not available" in out


def test_iteration_cap_stops_runaway_loop(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    runaway = [
        _tool_call_response("list_ideas", {}) for _ in range(20)
    ]
    llm = MockLLMProvider(runaway)

    inputs = iter(["list ideas forever", "/exit"])
    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    out = console.file.getvalue()
    assert "too many tool iterations" in out.lower()


def test_tools_schema_advertised_to_llm(config_dir) -> None:
    """LLM should receive the OpenAI tools array on each chat_with_tools call."""
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)

    llm = MockLLMProvider([ChatResponse(content="ok")])
    inputs = iter(["hi", "/exit"])
    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    assert llm.tool_calls_seen, "chat_with_tools should have been invoked"
    tools = llm.tool_calls_seen[0]
    assert tools is not None
    tool_names = {t["function"]["name"] for t in tools}
    assert {"search_arxiv", "load_paper", "discuss_idea"}.issubset(tool_names)
