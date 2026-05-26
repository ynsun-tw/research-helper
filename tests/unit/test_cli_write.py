"""Unit tests for ``research write`` CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from research_agent.agents.memory_keeper import Association
from research_agent.agents.scribe import Scribe
from research_agent.cli import app
from research_agent.cli_write import (
    WritingContext,
    _build_writing_context,
    run_write,
)
from research_agent.config import Config
from research_agent.core.idea import Idea
from research_agent.core.llm import MockLLMProvider
from research_agent.storage.discussions import DiscussionMessage
from research_agent.style.fingerprint import Fingerprint, MicroFingerprint


class _StubMemoryKeeper:
    """Drop-in stand-in for MemoryKeeper that doesn't need a real DB / Chroma."""

    def __init__(
        self,
        *,
        associations: list[Association] | None = None,
        history: list[DiscussionMessage] | None = None,
        raise_on_associations: bool = False,
        raise_on_history: bool = False,
    ) -> None:
        self._assocs = associations or []
        self._history = history or []
        self._raise_assoc = raise_on_associations
        self._raise_hist = raise_on_history
        self.assoc_calls: list[str] = []
        self.history_calls: list[str] = []

    def check_associations(
        self,
        context: str,
        *,
        threshold: float = 0.5,
        limit: int = 3,
        statuses=(),
    ) -> list[Association]:
        self.assoc_calls.append(context)
        if self._raise_assoc:
            raise RuntimeError("boom")
        return self._assocs[:limit]

    def recall_history(self, query: str, *, limit: int = 3) -> list[DiscussionMessage]:
        self.history_calls.append(query)
        if self._raise_hist:
            raise RuntimeError("boom")
        return self._history[:limit]

runner = CliRunner()


def _enqueue(mock: MockLLMProvider, n: int) -> None:
    for i in range(n):
        mock.enqueue(
            json.dumps(
                {
                    "draft": f"Draft {i} body text spanning multiple sentences for tests.",
                    "style_note": f"Voice variant {i}.",
                }
            )
        )


def test_write_without_fingerprint(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    mock = MockLLMProvider()
    _enqueue(mock, 3)
    scribe = Scribe(mock)
    result = run_write(
        cfg,
        Console(),
        section="abstract",
        versions=3,
        target_words=150,
        parallel=False,
        scribe=scribe,
    )
    assert result.section == "abstract"
    assert len(result.drafts) == 3
    # No fingerprint at start; cli should not crash.
    assert not cfg.fingerprint_path.exists()


def test_write_loads_existing_fingerprint(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    fp = Fingerprint(
        micro=MicroFingerprint(
            avg_sentence_length=18.0,
            sentence_count=10,
        ),
        sample_count=5,
        paper_count=1,
    )
    fp.save_to(cfg.fingerprint_path)

    mock = MockLLMProvider()
    _enqueue(mock, 1)
    scribe = Scribe(mock)
    result = run_write(
        cfg,
        Console(),
        section="introduction",
        versions=1,
        target_words=200,
        parallel=False,
        scribe=scribe,
    )
    assert len(result.drafts) == 1
    # The prompt the Scribe saw must include fingerprint context (sentence stats).
    prompts = [m[-1].content for m in mock.calls]
    assert any("sentence length" in p for p in prompts)


def test_write_persists_to_output_file(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    mock = MockLLMProvider()
    _enqueue(mock, 2)
    scribe = Scribe(mock)
    out = tmp_path / "drafts.md"
    result = run_write(
        cfg,
        Console(),
        section="conclusion",
        versions=2,
        output=out,
        parallel=False,
        scribe=scribe,
    )
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Scribe drafts — conclusion" in content
    assert "Version A" in content
    assert "Version B" in content
    assert len(result.drafts) == 2


def test_write_handles_broken_fingerprint(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    cfg = Config.load(config_dir)
    cfg.style_dir.mkdir(parents=True, exist_ok=True)
    cfg.fingerprint_path.write_text("not valid json", encoding="utf-8")

    mock = MockLLMProvider()
    _enqueue(mock, 1)
    scribe = Scribe(mock)
    # Should warn but not raise; we still get drafts back.
    result = run_write(
        cfg,
        Console(),
        section="abstract",
        versions=1,
        parallel=False,
        scribe=scribe,
    )
    assert len(result.drafts) == 1


def test_write_help() -> None:
    result = runner.invoke(app, ["write", "--help"])
    assert result.exit_code == 0
    assert "section" in result.stdout.lower()
    assert "versions" in result.stdout.lower() or "-n" in result.stdout


# --- S4.2.2: context-aware writing ------------------------------------------


def _idea(title: str, desc: str, *, status: str = "active", score: float | None = None) -> Idea:
    return Idea(
        id="aaaaaaaa-0000-0000-0000-000000000000",
        title=title,
        description=desc,
        status=status,  # type: ignore[arg-type]
        critic_score=score,
    )


def test_writing_context_render_empty() -> None:
    assert WritingContext().render() == ""
    assert WritingContext().is_empty()


def test_writing_context_render_full() -> None:
    ctx = WritingContext(
        user_text="sparse top-k attention for 32k contexts",
        related_ideas=[
            _idea(
                "Sparse routing attention",
                "Pick K experts per token",
                status="shelved",
                score=6.0,
            ),
        ],
        recent_discussions=[
            DiscussionMessage(
                id="m1",
                session_id="s1",
                role="user",
                content="What's the latency floor for top-k?",
            ),
        ],
        existing_drafts=[("intro.md", "# Existing intro paragraph...")],
    )
    rendered = ctx.render()
    assert "[USER CONTEXT]" in rendered
    assert "sparse top-k attention" in rendered
    assert "[RELATED IDEAS FROM YOUR LIBRARY]" in rendered
    assert "Sparse routing attention" in rendered
    assert "[RECENT DISCUSSION EXCERPTS]" in rendered
    assert "latency floor" in rendered
    assert "[EXISTING DRAFTS TO STAY CONSISTENT WITH" in rendered
    assert "intro.md" in rendered


def test_writing_context_truncates_long_existing_draft(tmp_path: Path) -> None:
    long_body = "x " * 5000
    ctx = WritingContext(existing_drafts=[("big.md", long_body)])
    rendered = ctx.render()
    assert "[... truncated ...]" in rendered
    # Plus the rendered output shouldn't be unbounded
    assert len(rendered) < 2500


def test_build_writing_context_pulls_from_memory(tmp_path: Path) -> None:
    keeper = _StubMemoryKeeper(
        associations=[
            Association(idea=_idea("Sparse routing", "Top-K experts"), similarity=0.7),
        ],
        history=[
            DiscussionMessage(
                id="m1", session_id="s1", role="analyst", content="Recent debate notes."
            ),
        ],
    )
    ctx = _build_writing_context(
        user_text="long-context attention",
        check_against=None,
        memory_keeper=keeper,  # type: ignore[arg-type]
    )
    assert ctx.related_ideas
    assert ctx.recent_discussions
    assert keeper.assoc_calls == ["long-context attention"]
    assert keeper.history_calls == ["long-context attention"]


def test_build_writing_context_no_memory_when_empty_user_text() -> None:
    keeper = _StubMemoryKeeper(
        associations=[Association(idea=_idea("X", "y"), similarity=0.7)],
    )
    ctx = _build_writing_context(
        user_text="",
        check_against=None,
        memory_keeper=keeper,  # type: ignore[arg-type]
    )
    assert ctx.related_ideas == []
    assert ctx.recent_discussions == []
    assert keeper.assoc_calls == []
    assert keeper.history_calls == []


def test_build_writing_context_check_against_reads_file(tmp_path: Path) -> None:
    file_a = tmp_path / "intro.md"
    file_a.write_text("# Intro\n\nExisting intro content.", encoding="utf-8")
    ctx = _build_writing_context(
        user_text="",
        check_against=[file_a],
        memory_keeper=None,
    )
    assert ctx.existing_drafts
    assert ctx.existing_drafts[0][0] == "intro.md"
    assert "Existing intro content" in ctx.existing_drafts[0][1]


def test_build_writing_context_check_against_missing_file_is_skipped(
    tmp_path: Path,
) -> None:
    ctx = _build_writing_context(
        user_text="",
        check_against=[tmp_path / "missing.md"],
        memory_keeper=None,
        console=Console(),
    )
    assert ctx.existing_drafts == []


def test_build_writing_context_swallows_memory_errors() -> None:
    keeper = _StubMemoryKeeper(raise_on_associations=True, raise_on_history=True)
    ctx = _build_writing_context(
        user_text="some context",
        check_against=None,
        memory_keeper=keeper,  # type: ignore[arg-type]
        console=Console(),
    )
    # Errors must not crash the build; degrade to empty memory slots.
    assert ctx.related_ideas == []
    assert ctx.recent_discussions == []


def test_run_write_uses_memory_keeper_when_provided(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    mock = MockLLMProvider()
    _enqueue(mock, 1)
    scribe = Scribe(mock)
    keeper = _StubMemoryKeeper(
        associations=[Association(idea=_idea("Idea X", "desc Y"), similarity=0.7)],
        history=[DiscussionMessage(id="m1", session_id="s1", role="user", content="foo bar")],
    )
    result = run_write(
        cfg,
        Console(),
        section="introduction",
        context="long-context attention",
        versions=1,
        parallel=False,
        scribe=scribe,
        memory_keeper=keeper,  # type: ignore[arg-type]
    )
    assert result.context is not None
    assert result.context.related_ideas
    assert result.context.recent_discussions
    prompts = [m[-1].content for m in mock.calls]
    assert any("[RELATED IDEAS" in p for p in prompts)
    assert any("[RECENT DISCUSSION EXCERPTS" in p for p in prompts)
    assert any("[USER CONTEXT]" in p for p in prompts)


def test_run_write_check_against_propagates(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    draft_file = tmp_path / "abstract.md"
    draft_file.write_text("# Abstract\n\nOur existing abstract.", encoding="utf-8")

    mock = MockLLMProvider()
    _enqueue(mock, 1)
    scribe = Scribe(mock)
    result = run_write(
        cfg,
        Console(),
        section="introduction",
        check_against=[draft_file],
        versions=1,
        parallel=False,
        scribe=scribe,
    )
    assert result.context is not None
    assert result.context.existing_drafts
    prompts = [m[-1].content for m in mock.calls]
    assert any("EXISTING DRAFTS" in p and "Our existing abstract" in p for p in prompts)
