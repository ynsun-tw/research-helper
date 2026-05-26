"""Tests for the conversational REPL router and slash dispatch."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

import research_agent.chat.tools as chat_tools
from research_agent.chat import run_chat
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
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


def test_help_lists_slash_commands(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([])
    inputs = iter(["/help", "/exit"])

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
    assert "/read" in out
    assert "/discuss" in out
    assert "Session saved" in out


def test_unknown_slash_command(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([])
    inputs = iter(["/notacommand", "/exit"])

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    out = console.file.getvalue()
    assert "Unknown command" in out


def test_discuss_without_anchor_paper(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([])
    inputs = iter(["/discuss apply to biology", "/exit"])

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    out = console.file.getvalue()
    assert "No anchor paper" in out


def test_natural_language_routes_to_llm(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
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
    assert llm.calls, "LLM should have been called for plain text input"
    assert llm.calls[0][0].role == "system"


def test_read_then_discuss_then_paper(
    config_dir,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(
        [ANALYST_JSON, CRITIC_JSON, IDEA_ANALYST_JSON, IDEA_CRITIC_JSON]
    )

    monkeypatch.setattr(
        chat_tools,
        "_load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    inputs = iter(
        [
            "/read arxiv:1706.03762",
            "/paper",
            "/discuss Apply attention to biology",
            "/exit",
        ]
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
    assert "Analyst" in out
    assert "Critic" in out
    assert "Attention Is All You Need" in out
    assert "Session saved" in out


def test_idea_save_persists_after_discuss(
    config_dir,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(
        [ANALYST_JSON, CRITIC_JSON, IDEA_ANALYST_JSON, IDEA_CRITIC_JSON]
    )

    monkeypatch.setattr(
        chat_tools,
        "_load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    inputs = iter(
        [
            "/read arxiv:1706.03762",
            "/discuss Apply attention to biology",
            "/idea save Attention for biology",
            "/exit",
        ]
    )

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    out = console.file.getvalue()
    assert "Idea saved" in out

    from research_agent.storage.database import Database
    from research_agent.storage.ideas import IdeaRepository

    db = Database(cfg.db_path)
    try:
        repo = IdeaRepository(db)
        ideas = repo.list_all()
        assert len(ideas) == 1
        assert ideas[0].title == "Attention for biology"
    finally:
        db.close()


def test_session_close_persists_memory(config_dir) -> None:
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(["A short reply."])
    session = ChatSession.create(
        cfg=cfg,
        llm=llm,
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    session.memory.append("user", "hello")
    saved = session.close()
    assert saved == 1

    from research_agent.storage.database import Database
    from research_agent.storage.discussions import DiscussionRepository

    db = Database(cfg.db_path)
    try:
        repo = DiscussionRepository(db)
        msgs = repo.list_session(session.memory.session_id)
        assert len(msgs) == 1
        assert msgs[0].content == "hello"
    finally:
        db.close()
