"""`/insights` slash + `research_insights` LLM tool (M3 T3.3.2.4)."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    _parse_since_days,
    cmd_insights,
    exec_research_insights,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


@pytest.fixture
def session(tmp_path) -> ChatSession:
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    return ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )


def _console_out(session: ChatSession) -> str:
    file = session.console.file
    assert isinstance(file, StringIO)
    return file.getvalue()


def test_parse_since_days_all() -> None:
    assert _parse_since_days("") is None
    assert _parse_since_days("all") is None


def test_parse_since_days_numeric() -> None:
    assert _parse_since_days("30") == 30


def test_parse_since_days_units() -> None:
    assert _parse_since_days("7d") == 7
    assert _parse_since_days("2w") == 14
    assert _parse_since_days("6m") == 180
    assert _parse_since_days("1y") == 365


def test_parse_since_days_invalid() -> None:
    with pytest.raises(ValueError):
        _parse_since_days("yesterday")


def test_insights_slash_renders_markdown(session) -> None:
    # Seed a paper so the report has something to talk about.
    from research_agent.core.paper import Paper

    session.papers.save(
        Paper(
            id="arxiv:1",
            title="A study",
            authors=["X"],
            year=2024,
            venue="ICML",
            tags=["tag-one"],
        )
    )
    cmd_insights(session, "")
    out = _console_out(session)
    compact = " ".join(out.split())
    assert "Research Insights" in compact
    assert "Papers" in compact
    # Markdown shouldn't include the literal "[--since" usage hint when
    # called with no args.
    assert "Unknown flag" not in compact


def test_insights_slash_invalid_since(session) -> None:
    cmd_insights(session, "--since notatime")
    out = _console_out(session)
    assert "Error" in out


def test_insights_slash_unknown_flag(session) -> None:
    cmd_insights(session, "--bogus")
    assert "Unknown flag" in _console_out(session)


def test_insights_slash_accepts_since_flag(session) -> None:
    cmd_insights(session, "--since 30d")
    # No error markers; the report renders.
    assert "Error" not in _console_out(session)


def test_insights_slash_accepts_equals_form(session) -> None:
    cmd_insights(session, "--since=7d")
    assert "Error" not in _console_out(session)


def test_insights_writes_system_memory(session) -> None:
    cmd_insights(session, "")
    sys_msgs = [m for m in session.memory.messages if m.role == "system"]
    assert any("Insights report" in m.content for m in sys_msgs)


def test_exec_research_insights_default(session) -> None:
    out = exec_research_insights(session, {})
    assert "# Research Insights" in out


def test_exec_research_insights_with_since_days(session) -> None:
    out = exec_research_insights(session, {"since_days": 30})
    assert "last 30 days" in out


def test_exec_research_insights_rejects_bad_since_days(session) -> None:
    out = exec_research_insights(session, {"since_days": "abc"})
    assert "Error" in out
