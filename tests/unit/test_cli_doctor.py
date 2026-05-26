"""Unit tests for ``research doctor`` (M5 S5.4.2)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.cli_doctor import (
    CheckResult,
    _check_api_key,
    _check_config_file,
    _check_data_dir,
    _check_database,
    _check_disk_space,
    _check_package_version,
    _render,
    run_doctor,
    safe_load,
)
from research_agent.config import Config

# --- per-check unit tests ---------------------------------------------------


def _cfg_for(tmp_path: Path, **overrides) -> Config:
    cfg = Config(data_dir=tmp_path, **overrides)
    return cfg


def test_check_api_key_missing(tmp_path: Path) -> None:
    result = _check_api_key(_cfg_for(tmp_path, api_key=None))
    assert result.status == "fail"
    assert "no api_key" in result.message


def test_check_api_key_openrouter_prefix_ok(tmp_path: Path) -> None:
    result = _check_api_key(_cfg_for(tmp_path, api_key="sk-or-abc123"))
    assert result.status == "ok"


def test_check_api_key_unknown_prefix_warns(tmp_path: Path) -> None:
    result = _check_api_key(_cfg_for(tmp_path, api_key="custom-token-zz"))
    assert result.status == "warn"


def test_check_config_file_missing(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    result = _check_config_file(cfg)
    assert result.status == "warn"
    assert "no file at" in result.message


def test_check_config_file_wrong_mode(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text("api_key: x\n", encoding="utf-8")
    cfg.config_path.chmod(0o644)
    result = _check_config_file(cfg)
    assert result.status == "warn"
    assert "chmod 600" in result.hint
    assert "0o600" in result.message


def test_check_config_file_correct_mode(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text("api_key: x\n", encoding="utf-8")
    cfg.config_path.chmod(0o600)
    result = _check_config_file(cfg)
    assert result.status == "ok"


def test_check_data_dir_missing_warns(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path / "does-not-exist")
    result = _check_data_dir(cfg)
    assert result.status == "warn"


def test_check_data_dir_is_file_fails(tmp_path: Path) -> None:
    fake = tmp_path / "conflict"
    fake.write_text("not a dir", encoding="utf-8")
    cfg = _cfg_for(fake)
    result = _check_data_dir(cfg)
    assert result.status == "fail"


def test_check_database_missing_is_ok(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    result = _check_database(cfg)
    assert result.status == "ok"


def test_check_database_integrity_ok(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.db_path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()
    result = _check_database(cfg)
    assert result.status == "ok"


def test_check_database_corrupt_fails(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Bytes that look like a sqlite header but aren't valid → integrity_check
    # would still say "ok" though, because sqlite is forgiving. Easier:
    # write nonsense and check that connect itself fails on a query.
    cfg.db_path.write_bytes(b"this is not a sqlite database")
    result = _check_database(cfg)
    assert result.status == "fail"
    assert "unreadable" in result.message or "integrity" in result.message


def test_check_disk_space_returns_some_status(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path)
    result = _check_disk_space(cfg)
    assert result.status in ("ok", "warn", "fail")
    assert "MB free" in result.message or "could not stat" in result.message


def test_check_package_version_reports_paper_research_agent() -> None:
    result = _check_package_version()
    # In dev environments the package is always installed editably,
    # so we expect ok. CI installs from sdist or wheel — same story.
    assert result.status in ("ok", "warn")
    if result.status == "ok":
        assert "paper-research-agent" in result.message


# --- run_doctor end-to-end --------------------------------------------------


def test_run_doctor_returns_zero_with_clean_config(tmp_path: Path) -> None:
    cfg = _cfg_for(tmp_path, api_key="sk-or-xyz")
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text("api_key: sk-or-xyz\n", encoding="utf-8")
    cfg.config_path.chmod(0o600)
    console = Console(record=True, width=120)
    rc = run_doctor(cfg, console)
    assert rc == 0
    text = console.export_text()
    assert "research doctor" in text
    assert "api key" in text


def test_run_doctor_returns_one_when_critical_check_fails(tmp_path: Path) -> None:
    # Missing api_key is a hard fail.
    cfg = _cfg_for(tmp_path, api_key=None)
    console = Console(record=True, width=120)
    rc = run_doctor(cfg, console)
    assert rc == 1
    assert "1 fail" in console.export_text()


def test_run_doctor_handles_none_config(tmp_path: Path) -> None:
    console = Console(record=True, width=120)
    rc = run_doctor(None, console)
    assert rc == 1
    text = console.export_text()
    assert "could not load" in text.lower()


# --- CLI integration --------------------------------------------------------


def test_cli_doctor_command_runs(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg_for(tmp_path, api_key="sk-or-xyz")
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text("api_key: sk-or-xyz\n", encoding="utf-8")
    cfg.config_path.chmod(0o600)

    def _fake_safe_load(console):
        return cfg

    monkeypatch.setattr(
        "research_agent.cli_doctor.safe_load", _fake_safe_load
    )

    runner = CliRunner()
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code in (0, 1)  # depending on disk / chromadb status
    assert "research doctor" in res.output


def test_cli_doctor_shown_in_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "doctor" in res.output


# --- safe_load fallback -----------------------------------------------------


def test_safe_load_returns_config_when_load_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg_for(tmp_path)
    monkeypatch.setattr(
        "research_agent.cli_doctor.Config.load", lambda *a, **k: cfg
    )
    console = Console(record=True, width=120)
    assert safe_load(console) is cfg


def test_safe_load_returns_none_when_load_raises(monkeypatch) -> None:
    def _boom(*_, **__):
        raise ValueError("simulated config crash")

    monkeypatch.setattr("research_agent.cli_doctor.Config.load", _boom)
    console = Console(record=True, width=120)
    assert safe_load(console) is None
    assert "simulated config crash" in console.export_text()


# --- _render UI smoke -------------------------------------------------------


def test_render_writes_status_summary() -> None:
    results = [
        CheckResult("a", "ok", "fine", ""),
        CheckResult("b", "warn", "watch out", "do x"),
        CheckResult("c", "fail", "broken", "fix it"),
    ]
    console = Console(record=True, width=120)
    _render(results, console)
    out = console.export_text()
    assert "1 fail" in out
    assert "1 warn" in out


def test_render_all_ok_path() -> None:
    console = Console(record=True, width=120)
    _render([CheckResult("a", "ok", "fine", "")], console)
    assert "All checks passed" in console.export_text()
