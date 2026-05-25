"""Unit tests for CLI commands."""

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


def test_read_without_api_key(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_agent.cli._load_config",
        lambda: Config.load(config_dir),
    )
    result = runner.invoke(app, ["read", "arxiv:2301.12345"])
    assert result.exit_code == 1
    assert "API key" in result.stdout
