"""E2E tests for ``research write`` and ``research figure`` (M5 S5.4.3).

These tests exercise the full CLI dispatch path with a mocked LLM
so the pipeline (Typer parse → cli_write / cli_figure → agent →
Rich render → optional file output) is verified without hitting the
network.

The M5 spec asks for E2E coverage of the five core commands
(read / search / discuss / write / reproduce). ``read`` and
``queue`` already have dedicated E2E files; ``search`` and
``discuss`` are exercised through the conversational shell tests.
``reproduce`` is deferred. This file covers the M4-introduced
writing surface so every shipped CLI subcommand has an E2E
regression guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Config

runner = CliRunner()


_SCRIBE_REPLY = json.dumps(
    {
        "draft": "This is a deterministic section draft.\n\nIt has two paragraphs.",
        "style_note": "Concise variant for the test.",
    }
)
_ILLUSTRATOR_REPLY = json.dumps(
    {
        "code": "\\begin{tikzpicture}\\node {hello};\\end{tikzpicture}",
        "style_label": "layered horizontal",
        "notes": "deterministic for tests",
        "suggested_use": "system overview",
    }
)


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated config + data directory rooted at tmp_path."""
    cfg = Config(data_dir=tmp_path, api_key="sk-or-test")
    cfg.save()
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)
    monkeypatch.setattr("research_agent.cli._ensure_api_key", lambda: cfg)
    return tmp_path


def test_write_section_emits_drafts(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``research write introduction`` runs end-to-end with a mock LLM."""
    from research_agent.core.llm import LLMClient, MockLLMProvider

    # Three variants → three canned responses (parallel via thread pool).
    mock = MockLLMProvider(responses=[_SCRIBE_REPLY, _SCRIBE_REPLY, _SCRIBE_REPLY])
    monkeypatch.setattr(LLMClient, "from_config", lambda cfg: mock)

    out_path = config_dir / "intro.md"
    result = runner.invoke(
        app,
        [
            "write",
            "introduction",
            "--versions", "3",
            "--words", "120",
            "--output", str(out_path),
            "--sequential",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Version A" in result.stdout
    assert "Version B" in result.stdout
    assert "Version C" in result.stdout
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "# Scribe drafts" in body
    assert "deterministic section draft" in body


def test_figure_architecture_emits_tikz(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``research figure --type architecture`` end-to-end."""
    from research_agent.core.llm import LLMClient, MockLLMProvider

    mock = MockLLMProvider(responses=[_ILLUSTRATOR_REPLY, _ILLUSTRATOR_REPLY])
    monkeypatch.setattr(LLMClient, "from_config", lambda cfg: mock)

    out_path = config_dir / "fig.md"
    result = runner.invoke(
        app,
        [
            "figure",
            "--type", "architecture",
            "--desc", "three layer encoder",
            "--versions", "2",
            "--output", str(out_path),
            "--sequential",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Version A" in result.stdout
    assert "architecture" in result.stdout
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "tikzpicture" in body
    assert "```latex" in body


def test_figure_result_with_verify_runs_subprocess(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--verify`` on a result figure runs the generated code in a subprocess."""
    from research_agent.core.llm import LLMClient, MockLLMProvider

    # Cheap, deterministic Python that won't import anything heavy.
    runnable = json.dumps(
        {
            "code": "x = 1 + 1\n",
            "style_label": "trivial",
            "notes": "no real plot, just runs",
            "suggested_use": "test only",
        }
    )
    mock = MockLLMProvider(responses=[runnable])
    monkeypatch.setattr(LLMClient, "from_config", lambda cfg: mock)

    result = runner.invoke(
        app,
        [
            "figure",
            "--type", "result",
            "--desc", "trivial chart",
            "--versions", "1",
            "--verify",
            "--sequential",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Version A ran successfully" in result.stdout


def test_figure_concept_emits_target_model(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``research figure --type concept`` surfaces the target_model field."""
    from research_agent.core.llm import LLMClient, MockLLMProvider

    payload = json.dumps(
        {
            "code": "clean schematic of attention",
            "style_label": "flat schematic",
            "notes": "for DALL-E 3",
            "suggested_use": "method intro",
            "target_model": "dalle3",
            "negative_prompt": "",
        }
    )
    mock = MockLLMProvider(responses=[payload])
    monkeypatch.setattr(LLMClient, "from_config", lambda cfg: mock)

    result = runner.invoke(
        app,
        [
            "figure",
            "--type", "concept",
            "--desc", "attention flow",
            "--versions", "1",
            "--sequential",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "target: dalle3" in result.stdout
