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
