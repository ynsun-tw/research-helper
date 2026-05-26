"""Unit tests for the CLI surface (REPL entry + config subcommand)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Config

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Research Agent" in result.stdout


def test_help_only_keeps_config_subcommand() -> None:
    """After M2.5 the legacy read/discuss/ideas subcommands are removed."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "config" in result.stdout
    # Removed subcommands should not appear in help.
    assert " read " not in result.stdout
    assert " discuss " not in result.stdout
    assert " ideas " not in result.stdout


def test_config_show_empty(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_agent.cli._load_config",
        lambda: Config.load(config_dir),
    )
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "(not set)" in result.stdout


def test_config_set_and_show(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("research_agent.cli._load_config", lambda: Config.load(config_dir))
    result = runner.invoke(app, ["config", "set", "api_key", "sk-testkey12345678"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "testkey12345678" not in result.stdout


def test_config_set_alert_threshold_persists(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_agent.cli._load_config", lambda: Config.load(config_dir))
    result = runner.invoke(app, ["config", "set", "alert_threshold", "0.65"])
    assert result.exit_code == 0, result.stdout
    # Round-trip through disk.
    reloaded = Config.load(config_dir)
    assert reloaded.alert_threshold == pytest.approx(0.65)


def test_config_set_alert_threshold_rejects_out_of_range(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_agent.cli._load_config", lambda: Config.load(config_dir))
    result = runner.invoke(app, ["config", "set", "alert_threshold", "1.5"])
    assert result.exit_code != 0
    assert "0.0" in result.stdout or "1.0" in result.stdout


def test_config_set_alert_threshold_rejects_non_numeric(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("research_agent.cli._load_config", lambda: Config.load(config_dir))
    result = runner.invoke(app, ["config", "set", "alert_threshold", "high"])
    assert result.exit_code != 0
    assert "number" in result.stdout.lower()


def test_config_show_includes_alert_threshold(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "research_agent.cli._load_config", lambda: Config.load(config_dir)
    )
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "alert_threshold" in result.stdout
    assert "0.80" in result.stdout  # default


def test_repl_without_api_key_errors(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default entry (REPL) should refuse to start without an API key."""
    monkeypatch.setattr(
        "research_agent.cli._load_config",
        lambda: Config.load(config_dir),
    )
    result = runner.invoke(app, [])
    assert result.exit_code == 1
    assert "API key" in result.stdout


def test_repl_starts_with_api_key(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When configured, ``research`` enters the chat shell and exits cleanly."""
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)

    from research_agent.core.llm import LLMClient, MockLLMProvider

    monkeypatch.setattr(
        LLMClient,
        "from_config",
        lambda config: MockLLMProvider([]),
    )

    result = runner.invoke(app, [], input="/exit\n")
    assert result.exit_code == 0, result.stdout
    assert "Research Agent" in result.stdout
    assert "Session saved" in result.stdout
