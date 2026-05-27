"""Phase 1 chat tools: doctor + style read-only.

Locks the public surface of the new tools so the LLM router contract
doesn't drift silently: slash commands render something, executors
return non-empty summary text that mentions the right state, and
empty / fresh installs degrade to helpful nudges instead of crashes.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    cmd_doctor,
    cmd_style,
    exec_run_doctor,
    exec_style_history,
    exec_style_show,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


@pytest.fixture
def session(tmp_path) -> ChatSession:
    cfg = Config(data_dir=tmp_path, api_key="sk-or-test")
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


# ---- doctor ---------------------------------------------------------------


def test_doctor_slash_renders_table(session: ChatSession) -> None:
    cmd_doctor(session, "")
    out = _console_out(session)
    assert "doctor" in out.lower()


def test_doctor_slash_writes_system_memory(session: ChatSession) -> None:
    cmd_doctor(session, "")
    sys_msgs = [m for m in session.memory.messages if m.role == "system"]
    assert any("[doctor]" in m.content for m in sys_msgs)


def test_exec_run_doctor_returns_summary(session: ChatSession) -> None:
    out = exec_run_doctor(session, {})
    assert "Environment health" in out
    assert "rendered to the user" in out


# ---- style (read-only) ----------------------------------------------------


def test_style_slash_default_shows(session: ChatSession) -> None:
    cmd_style(session, "")
    out = _console_out(session)
    assert "No style samples" in out or "Style corpus" in out


def test_style_slash_history_on_empty_install(session: ChatSession) -> None:
    cmd_style(session, "history")
    out = _console_out(session).lower()
    assert "no fingerprint" in out or "fingerprint" in out


def test_style_slash_unknown_sub(session: ChatSession) -> None:
    cmd_style(session, "bogus")
    assert "Unknown" in _console_out(session)


def test_exec_style_show_empty_corpus_nudges_user(session: ChatSession) -> None:
    out = exec_style_show(session, {})
    assert "empty" in out.lower() or "train_style" in out
    # No crash even when fingerprint dir doesn't exist.


def test_exec_style_history_empty_nudges_user(session: ChatSession) -> None:
    out = exec_style_history(session, {})
    assert "build_fingerprint" in out or "No fingerprint history" in out


def test_exec_style_show_with_samples_reports_counts(
    session: ChatSession,
) -> None:
    from research_agent.storage.database import Database
    from research_agent.style.samples import StyleSampleRepository

    db = Database(session.cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        repo.add(
            paper_id="arxiv:0000",
            section_title="Introduction",
            paragraph="One paragraph of our own writing for fingerprint training.",
            char_count=60,
            word_count=10,
            sentence_count=1,
        )
    finally:
        db.close()

    out = exec_style_show(session, {})
    assert "1 paragraph" in out
    assert "1 source paper" in out
    # No fingerprint yet → message should say so.
    assert "NOT built" in out


# ---- LLM tool registry sanity ---------------------------------------------


def test_new_tools_are_registered_for_llm() -> None:
    from research_agent.chat.tools import LLM_TOOLS

    assert "run_doctor" in LLM_TOOLS
    assert "style_show" in LLM_TOOLS
    assert "style_history" in LLM_TOOLS


def test_new_slashes_are_registered() -> None:
    from research_agent.chat.tools import SLASH_COMMANDS

    assert "doctor" in SLASH_COMMANDS
    assert "style" in SLASH_COMMANDS
