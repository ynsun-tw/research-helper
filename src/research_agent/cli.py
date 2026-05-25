"""Typer CLI entry point for Research Agent."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research_agent.cli_services import run_discuss, run_read
from research_agent.config import Config, ConfigError
from research_agent.core.language import language_label
from research_agent.core.llm import LLMClient

app = typer.Typer(
    name="research",
    help="Research Agent — local-first multi-agent CLI for researchers.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage configuration (API keys, model, paths).")
app.add_typer(config_app, name="config")

console = Console()


def _load_config() -> Config:
    return Config.load()


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Config key (api_key, model, language, base_url, app_title, app_url, data_dir)",
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
    table.add_row("config_path", str(cfg.config_path))
    console.print(table)


@app.command()
def read(
    source: str = typer.Argument(
        ...,
        help="Paper source: arxiv:ID (e.g. arxiv:2301.12345) or local PDF path",
    ),
) -> None:
    """Analyze a paper with Analyst + Critic dual perspectives."""
    cfg = _ensure_api_key()
    llm = LLMClient.from_config(cfg)
    code = run_read(source, cfg, llm, console)
    raise typer.Exit(code=code)


@app.command()
def discuss(
    topic: str | None = typer.Option(
        None,
        "--topic",
        "-t",
        help="Optional opening topic for the discussion",
    ),
) -> None:
    """Start an interactive critical discussion (Analyst + Critic each turn)."""
    cfg = _ensure_api_key()
    llm = LLMClient.from_config(cfg)
    code = run_discuss(cfg, llm, console, opening_topic=topic)
    raise typer.Exit(code=code)


def _ensure_api_key() -> Config:
    """Validate API key before commands that need LLM access."""
    cfg = _load_config()
    try:
        cfg.require_api_key()
    except ConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    return cfg


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    main()
