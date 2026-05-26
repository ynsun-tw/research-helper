"""Unit tests for ``research write`` CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from research_agent.agents.scribe import Scribe
from research_agent.cli import app
from research_agent.cli_write import run_write
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.style.fingerprint import Fingerprint, MicroFingerprint

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
