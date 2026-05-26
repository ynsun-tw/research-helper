"""Unit tests for ``research review`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from research_agent.agents.orchestrator import Orchestrator
from research_agent.agents.scribe import Scribe
from research_agent.cli import app
from research_agent.cli_review import run_review
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.storage.database import Database
from research_agent.storage.draft_revisions import DraftRevisionRepository

runner = CliRunner()


def _enqueue_pipeline(mock: MockLLMProvider) -> None:
    mock.enqueue(json.dumps({"issues": ["analyst issue"], "summary": "ok"}))
    mock.enqueue(json.dumps({"issues": ["critic issue"], "summary": "watch claims"}))
    mock.enqueue(
        json.dumps(
            {
                "draft": "Revised body addressing both reviews.",
                "style_note": "Toned down overclaim, added related-work pointer.",
            }
        )
    )


def test_run_review_with_file_path(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    draft_path = tmp_path / "intro.md"
    draft_path.write_text(
        "First sentence of the intro. Second sentence claims too much.",
        encoding="utf-8",
    )
    mock = MockLLMProvider()
    _enqueue_pipeline(mock)
    orchestrator = Orchestrator(mock)
    scribe = Scribe(mock)

    result = run_review(
        cfg,
        Console(),
        draft_path=draft_path,
        section="introduction",
        orchestrator=orchestrator,
        scribe=scribe,
    )
    assert result.reviewed.analyst_review is not None
    assert result.reviewed.critic_review is not None
    assert "Revised body" in result.reviewed.revised.text
    assert len(mock.calls) == 3


def test_run_review_persists_output(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    out = tmp_path / "review.md"
    mock = MockLLMProvider()
    _enqueue_pipeline(mock)
    orchestrator = Orchestrator(mock)
    scribe = Scribe(mock)

    run_review(
        cfg,
        Console(),
        draft_text="Draft text under review here.",
        section="abstract",
        output=out,
        orchestrator=orchestrator,
        scribe=scribe,
    )
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "## Original" in body
    assert "## Analyst review" in body
    assert "## Critic review" in body
    assert "## Revised" in body
    assert "analyst issue" in body
    assert "critic issue" in body
    assert "Revised body" in body


def test_run_review_empty_draft_raises(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    try:
        run_review(cfg, Console(), draft_text="   ", section="introduction")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty draft")


def test_review_help() -> None:
    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    assert "section" in result.stdout.lower()
    assert "draft" in result.stdout.lower()
    assert "interactive" in result.stdout.lower()


# --- S4.3.2: interactive selection ------------------------------------------


def _make_prompt(answers: list[str]):
    """Build a prompt callable that returns each answer in turn."""
    it = iter(answers)

    def _prompt(_question: str) -> str:
        try:
            return next(it)
        except StopIteration:
            return ""

    return _prompt


def _enqueue_interactive(mock: MockLLMProvider) -> None:
    """Two reviews queued; revision queued separately because we choose
    when (or if) to fire it after the interactive selection."""
    mock.enqueue(
        json.dumps(
            {
                "issues": ["unsupported claim", "missing baseline"],
                "suggestions": ["cite Smith 2024"],
                "summary": "ok",
            }
        )
    )
    mock.enqueue(
        json.dumps(
            {
                "issues": ["overclaim"],
                "suggestions": ["soften 'consistently'"],
                "summary": "watch claims",
            }
        )
    )


def test_review_interactive_accepts_subset(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    mock = MockLLMProvider()
    _enqueue_interactive(mock)
    mock.enqueue(
        json.dumps(
            {
                "draft": "Revised body addressing the chosen items.",
                "style_note": "Accepted overclaim + missing baseline.",
            }
        )
    )
    orchestrator = Orchestrator(mock)
    scribe = Scribe(mock)
    answers = [
        # Analyst issues: unsupported claim, missing baseline
        "n",
        "y",
        # Analyst suggestion: cite Smith 2024
        "n",
        # Critic issue: overclaim
        "y",
        # Critic suggestion: soften 'consistently'
        "n",
    ]
    result = run_review(
        cfg,
        Console(),
        draft_text="Original body of the introduction. Two sentences.",
        section="introduction",
        interactive=True,
        save=False,
        prompt_fn=_make_prompt(answers),
        orchestrator=orchestrator,
        scribe=scribe,
    )
    assert result.selection is not None
    assert result.selection.selected_issues == ["missing baseline", "overclaim"]
    assert result.selection.rejected_issues == ["unsupported claim"]
    assert result.selection.selected_suggestions == []
    assert "cite Smith 2024" in result.selection.rejected_suggestions
    assert "soften 'consistently'" in result.selection.rejected_suggestions
    assert "Revised body" in result.reviewed.revised.text
    # The revision prompt must only mention the accepted items.
    rev_prompt = mock.calls[-1][-1].content
    assert "missing baseline" in rev_prompt
    assert "overclaim" in rev_prompt
    assert "unsupported claim" not in rev_prompt
    assert "cite Smith 2024" not in rev_prompt


def test_review_interactive_all_rejected_skips_llm_revision(
    tmp_path: Path, config_dir: Path
) -> None:
    cfg = Config.load(config_dir)
    mock = MockLLMProvider()
    _enqueue_interactive(mock)
    orchestrator = Orchestrator(mock)
    scribe = Scribe(mock)
    answers = ["n"] * 10  # reject everything
    result = run_review(
        cfg,
        Console(),
        draft_text="Body.",
        section="introduction",
        interactive=True,
        save=False,
        prompt_fn=_make_prompt(answers),
        orchestrator=orchestrator,
        scribe=scribe,
    )
    # No items selected → scribe.revise short-circuits, no third LLM call.
    assert len(mock.calls) == 2
    assert result.reviewed.revised.text == result.reviewed.original.text
    assert result.reviewed.revised.style_note.startswith("No actionable")


def test_review_save_persists_to_repository(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    db = Database(cfg.db_path)
    repo = DraftRevisionRepository(db)
    try:
        mock = MockLLMProvider()
        _enqueue_pipeline(mock)
        orchestrator = Orchestrator(mock)
        scribe = Scribe(mock)
        result = run_review(
            cfg,
            Console(),
            draft_text="Original body to be revised soon.",
            section="introduction",
            save=True,
            orchestrator=orchestrator,
            scribe=scribe,
            revisions_repo=repo,
        )
        assert result.saved_revision is not None
        rows = repo.list_all()
        assert len(rows) == 1
        assert rows[0].original_text.startswith("Original body")
        assert "Revised body" in rows[0].revised_text
        # Not interactive in this path
        assert rows[0].interactive is False
    finally:
        db.close()


def test_review_save_skipped_when_revision_unchanged(
    tmp_path: Path, config_dir: Path
) -> None:
    """If Scribe.revise short-circuits (empty reviews), original == revised
    and we must not pollute the corpus with a no-op row."""
    cfg = Config.load(config_dir)
    db = Database(cfg.db_path)
    repo = DraftRevisionRepository(db)
    try:
        mock = MockLLMProvider()
        # Empty reviews → revise is short-circuited.
        mock.enqueue(json.dumps({"issues": [], "suggestions": [], "summary": ""}))
        mock.enqueue(json.dumps({"issues": [], "suggestions": [], "summary": ""}))
        orchestrator = Orchestrator(mock)
        scribe = Scribe(mock)
        result = run_review(
            cfg,
            Console(),
            draft_text="Already polished body of the draft.",
            section="introduction",
            save=True,
            orchestrator=orchestrator,
            scribe=scribe,
            revisions_repo=repo,
        )
        assert result.saved_revision is None
        assert repo.count() == 0
    finally:
        db.close()
