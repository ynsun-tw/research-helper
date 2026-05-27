"""``research doctor`` — environment health checks (M5 S5.4.2).

A diagnostic command users can run when something feels off. Each
check returns a triplet (status, message, hint) so the table reads
the same in every locale: a coloured glyph, a one-line statement,
and an actionable next step. The command exits non-zero if any
check is at ``"fail"`` so it slots into CI / install scripts.

The checks intentionally avoid LLM round-trips — they verify the
plumbing the user controls (config file mode, DB integrity, disk
budget, package metadata, ChromaDB availability) rather than the
upstream API. Use ``research config show`` followed by an actual
``research`` invocation to validate the API key end-to-end.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

from research_agent.config import Config, ConfigError

Status = Literal["ok", "warn", "fail"]

# Disk-space heuristic. ChromaDB grows linearly with embeddings and
# PDF cache can spike. < 500 MB free → warn; < 100 MB → fail.
DISK_FAIL_MB = 100
DISK_WARN_MB = 500


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    hint: str = ""


_STATUS_GLYPH: dict[Status, str] = {
    "ok": "[green]✓[/green]",
    "warn": "[yellow]![/yellow]",
    "fail": "[red]✗[/red]",
}


def run_doctor(cfg: Config | None, console: Console) -> int:
    """Run every health check and render a summary table.

    ``cfg`` may be ``None`` if config loading itself failed — in
    that case the first row of the table records the failure and
    every downstream check is skipped.
    """
    results: list[CheckResult] = []

    # Always start with config so the rest of the report has something
    # to lean on.
    if cfg is None:
        results.append(
            CheckResult(
                name="config",
                status="fail",
                message="could not load ~/.research-agent/config.yaml",
                hint="run `research config set api_key <key>` to create one",
            )
        )
    else:
        results.append(_check_config_file(cfg))
        results.append(_check_api_key(cfg))
        results.append(_check_data_dir(cfg))
        results.append(_check_database(cfg))
        results.append(_check_chroma_dir(cfg))
        results.append(_check_disk_space(cfg))
    results.append(_check_package_version())
    results.append(_check_chromadb_import())

    _render(results, console)

    # Exit code: 0 if all checks pass / only warn; 1 if any fail.
    return 0 if all(r.status != "fail" for r in results) else 1


# --- individual checks ------------------------------------------------------


def _check_config_file(cfg: Config) -> CheckResult:
    path = cfg.config_path
    if not path.exists():
        return CheckResult(
            "config file",
            "warn",
            f"no file at {path}",
            "run `research config set api_key <key>` to create one",
        )
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        return CheckResult(
            "config file",
            "warn",
            f"{path} has mode {oct(mode)} (expected 0o600)",
            f"run: chmod 600 {path}",
        )
    return CheckResult("config file", "ok", f"{path} (mode 0o600)")


def _check_api_key(cfg: Config) -> CheckResult:
    if not cfg.api_key:
        return CheckResult(
            "api key",
            "fail",
            "no api_key configured",
            "run `research config set api_key <openrouter-key>`",
        )
    masked = cfg.masked_api_key() if hasattr(cfg, "masked_api_key") else "set"
    # Heuristic only: the spec doesn't require a particular prefix
    # since users may proxy to any OpenAI-compatible endpoint.
    if cfg.api_key.startswith("sk-or-") or cfg.api_key.startswith("sk-"):
        return CheckResult("api key", "ok", f"set ({masked})")
    return CheckResult(
        "api key",
        "warn",
        f"set ({masked}) but doesn't match the usual OpenRouter / OpenAI prefix",
        "double-check the key if `research` fails with auth errors",
    )


def _check_data_dir(cfg: Config) -> CheckResult:
    path = cfg.data_dir
    if not path.exists():
        return CheckResult(
            "data dir",
            "warn",
            f"{path} does not exist yet (will be created on first use)",
        )
    if not path.is_dir():
        return CheckResult(
            "data dir",
            "fail",
            f"{path} is not a directory",
            "remove or rename the conflicting file",
        )
    return CheckResult("data dir", "ok", f"{path}")


def _check_database(cfg: Config) -> CheckResult:
    path = cfg.db_path
    if not path.exists():
        return CheckResult(
            "database",
            "ok",
            f"{path} (will be created on first use)",
        )
    try:
        conn = sqlite3.connect(path)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return CheckResult(
            "database",
            "fail",
            f"{path} is unreadable: {exc}",
            "back up the file and run `research config set data_dir <new-path>`",
        )
    integrity = (result[0] if result else "").lower()
    if integrity != "ok":
        return CheckResult(
            "database",
            "fail",
            f"{path} failed PRAGMA integrity_check: {integrity}",
            "restore from backup; SQLite cannot self-heal corruption",
        )
    return CheckResult("database", "ok", f"{path} (integrity OK)")


def _check_chroma_dir(cfg: Config) -> CheckResult:
    path = cfg.chroma_dir
    if not path.exists():
        return CheckResult(
            "chroma dir",
            "ok",
            f"{path} (will be created on first vector write)",
        )
    if not path.is_dir():
        return CheckResult(
            "chroma dir",
            "fail",
            f"{path} is not a directory",
            "remove the conflicting file",
        )
    # ChromaDB v0.5 stores the index under `chroma.sqlite3`; treat
    # absence as "no embeddings yet" rather than failure.
    return CheckResult("chroma dir", "ok", f"{path}")


def _check_disk_space(cfg: Config) -> CheckResult:
    path = cfg.data_dir if cfg.data_dir.exists() else cfg.data_dir.parent
    if not path.exists():
        path = Path.home()
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return CheckResult(
            "disk space",
            "warn",
            f"could not stat {path}: {exc}",
        )
    free_mb = usage.free // (1024 * 1024)
    if free_mb < DISK_FAIL_MB:
        return CheckResult(
            "disk space",
            "fail",
            f"only {free_mb} MB free at {path}",
            "free up space; ChromaDB and the PDF cache will spike",
        )
    if free_mb < DISK_WARN_MB:
        return CheckResult(
            "disk space",
            "warn",
            f"{free_mb} MB free at {path}",
            "consider freeing space before bulk paper imports",
        )
    return CheckResult("disk space", "ok", f"{free_mb} MB free at {path}")


def _check_package_version() -> CheckResult:
    """Read the installed distribution version (works for both dev
    installs via ``pip install -e .`` and wheel installs)."""
    try:
        version = importlib.metadata.version("paper-research-agent")
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(
            "package",
            "warn",
            "paper-research-agent is not installed via pip",
            "run `pip install -e .` from the project root",
        )
    return CheckResult("package", "ok", f"paper-research-agent {version}")


def _check_chromadb_import() -> CheckResult:
    """Confirm the heavy optional vector backend imports.

    ChromaDB is pulled in lazily by the REPL; doctor surfaces import
    errors up front so users don't first encounter them mid-session.
    """
    try:
        importlib.import_module("chromadb")
    except ImportError as exc:
        return CheckResult(
            "chromadb",
            "fail",
            f"chromadb import failed: {exc}",
            "reinstall: `pip install --force-reinstall chromadb`",
        )
    return CheckResult("chromadb", "ok", "imports cleanly")


# --- rendering --------------------------------------------------------------


def _render(results: list[CheckResult], console: Console) -> None:
    table = Table(
        title="research doctor", show_header=True, header_style="bold"
    )
    table.add_column("", width=2)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Hint", style="dim")
    for r in results:
        table.add_row(
            _STATUS_GLYPH[r.status],
            r.name,
            r.message,
            r.hint,
        )
    console.print(table)

    fails = sum(1 for r in results if r.status == "fail")
    warns = sum(1 for r in results if r.status == "warn")
    if fails:
        console.print(
            f"[red]✗ {fails} fail(s)[/red]" + (
                f", [yellow]{warns} warn(s)[/yellow]" if warns else ""
            )
        )
    elif warns:
        console.print(f"[yellow]! {warns} warn(s)[/yellow] (non-fatal)")
    else:
        console.print("[green]All checks passed.[/green]")


# --- safe entry point -------------------------------------------------------


def safe_load(console: Console) -> Config | None:
    """Best-effort ``Config.load`` for the doctor command.

    Doctor must run even when config loading itself errors out —
    that's precisely the situation users invoke it for. Return
    ``None`` and let the caller record the failure.
    """
    try:
        return Config.load()
    except (ConfigError, OSError, ValueError) as exc:
        console.print(f"[red]Could not load config:[/red] {exc}")
        return None
