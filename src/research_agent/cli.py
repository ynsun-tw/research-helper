"""Typer CLI entry point for Research Agent (conversational shell).

Only two surfaces remain after M2.5:
- ``research`` (no subcommand) enters the conversational REPL.
- ``research config ...`` manages persisted configuration.

Legacy ``read``/``discuss``/``ideas`` subcommands have been removed in favor of
the in-REPL slash commands (``/read``, ``/discuss``, ``/ideas``). The
underlying service functions (``run_read``, ``run_discuss``, ``run_ideas_*``)
remain importable so the chat tools and tests can call them directly.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research_agent.chat import run_chat
from research_agent.config import Config, ConfigError
from research_agent.core.language import language_label
from research_agent.core.llm import LLMClient

app = typer.Typer(
    name="research",
    help=(
        "Research Agent — conversational CLI for paper understanding. "
        "Run `research` to enter the chat shell."
    ),
    no_args_is_help=False,
    invoke_without_command=True,
)
config_app = typer.Typer(help="Manage configuration (API keys, model, paths).")
app.add_typer(config_app, name="config")

console = Console()


def _load_config() -> Config:
    return Config.load()


def _ensure_api_key() -> Config:
    """Validate API key before commands that need LLM access."""
    cfg = _load_config()
    try:
        cfg.require_api_key()
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    return cfg


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Default: enter the conversational REPL when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    cfg = _ensure_api_key()
    llm = LLMClient.from_config(cfg)
    code = run_chat(cfg, llm, console)
    raise typer.Exit(code=code)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help=(
            "Config key (api_key, model, language, base_url, app_title, "
            "app_url, data_dir, alert_threshold)"
        ),
    ),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Save a configuration value to ~/.research-agent/config.yaml."""
    cfg = _load_config()
    try:
        cfg.set_field(key, value)
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]✓[/green] Set [bold]{key}[/bold]")


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key to read"),
) -> None:
    """Print a single configuration value (api_key is masked)."""
    cfg = _load_config()
    try:
        console.print(cfg.get_field(key))
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@config_app.command("show")
def config_show() -> None:
    """Display current configuration (API key is masked)."""
    cfg = _load_config()
    table = Table(title="Research Agent Configuration", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("api_key", cfg.masked_api_key())
    table.add_row("model", cfg.model)
    table.add_row("language", f"{cfg.language} ({language_label(cfg.language)})")
    table.add_row("base_url", cfg.base_url)
    table.add_row("app_title", cfg.app_title)
    table.add_row("app_url", cfg.app_url)
    table.add_row("data_dir", str(cfg.data_dir))
    table.add_row("alert_threshold", f"{cfg.alert_threshold:.2f}")
    table.add_row("config_path", str(cfg.config_path))
    console.print(table)


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    main()
