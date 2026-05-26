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


def test_insights_subcommand_runs(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`research insights` should run against an empty DB and emit Markdown."""
    monkeypatch.setattr(
        "research_agent.cli._load_config",
        lambda: Config.load(config_dir),
    )
    result = runner.invoke(app, ["insights"])
    assert result.exit_code == 0, result.stdout
    assert "Research Insights" in result.stdout
    assert "Papers" in result.stdout


def test_insights_writes_output_file(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "research_agent.cli._load_config",
        lambda: Config.load(config_dir),
    )
    out_file = tmp_path / "report.md"
    result = runner.invoke(
        app, ["insights", "--since", "30d", "--output", str(out_file)]
    )
    assert result.exit_code == 0, result.stdout
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Research Insights" in content
    assert "last 30 days" in content


def test_insights_rejects_bad_since(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "research_agent.cli._load_config",
        lambda: Config.load(config_dir),
    )
    result = runner.invoke(app, ["insights", "--since", "yesterday"])
    assert result.exit_code == 1


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


# --- M5 S5.4.4 --version flag + version sync -------------------------------


def test_version_flag_long() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "research-agent" in result.stdout


def test_version_flag_short() -> None:
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "research-agent" in result.stdout


def test_version_matches_pyproject() -> None:
    """``research_agent.__version__`` must mirror the [project] version
    in ``pyproject.toml`` so the CLI and the distribution don't drift."""
    import tomllib

    from research_agent import __version__

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


# --- M5 S5.4.2 global error handler ----------------------------------------


def test_main_wraps_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random Python exceptions should be presented as one friendly line."""
    from research_agent import cli as cli_module

    def _boom() -> None:
        raise RuntimeError("simulated explosion")

    monkeypatch.setattr(cli_module, "app", _boom)
    monkeypatch.delenv("RESEARCH_AGENT_DEBUG", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()
    assert excinfo.value.code == 1


def test_main_propagates_with_debug_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """RESEARCH_AGENT_DEBUG=1 should bypass the wrapper for real debugging."""
    from research_agent import cli as cli_module

    def _boom() -> None:
        raise RuntimeError("simulated explosion")

    monkeypatch.setattr(cli_module, "app", _boom)
    monkeypatch.setenv("RESEARCH_AGENT_DEBUG", "1")
    with pytest.raises(RuntimeError, match="simulated explosion"):
        cli_module.main()


def test_main_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C should exit 130 with a one-line "Interrupted." note."""
    from research_agent import cli as cli_module

    def _interrupt() -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli_module, "app", _interrupt)
    monkeypatch.delenv("RESEARCH_AGENT_DEBUG", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()
    assert excinfo.value.code == 130


def test_main_handles_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ConfigError should produce a friendly message + hint."""
    from research_agent import cli as cli_module
    from research_agent.config import ConfigError

    def _broken_config() -> None:
        raise ConfigError("invalid yaml at line 7")

    monkeypatch.setattr(cli_module, "app", _broken_config)
    monkeypatch.delenv("RESEARCH_AGENT_DEBUG", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()
    assert excinfo.value.code == 1


def test_main_preserves_typer_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """``typer.Exit`` from a command must not be swallowed by the wrapper."""
    import typer

    from research_agent import cli as cli_module

    def _typer_exit() -> None:
        raise typer.Exit(code=2)

    monkeypatch.setattr(cli_module, "app", _typer_exit)
    monkeypatch.delenv("RESEARCH_AGENT_DEBUG", raising=False)
    with pytest.raises(typer.Exit) as excinfo:
        cli_module.main()
    assert excinfo.value.exit_code == 2
