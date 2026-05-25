"""Typer CLI entry point for Research Agent."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from research_agent.cli_ideas import run_ideas_list, run_ideas_show, run_ideas_update
from research_agent.cli_services import run_discuss, run_read
from research_agent.config import Config, ConfigError
from research_agent.core.idea import IDEA_STATUSES, IdeaStatus
from research_agent.core.language import language_label
from research_agent.core.llm import LLMClient

app = typer.Typer(
    name="research",
    help="Research Agent — local-first multi-agent CLI for researchers.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage configuration (API keys, model, paths).")
ideas_app = typer.Typer(help="Manage research ideas (list, show, update status).")
app.add_typer(config_app, name="config")
app.add_typer(ideas_app, name="ideas")

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
    paper: str = typer.Option(
        ...,
        "--paper",
        "-p",
        help="Anchor paper: arXiv id, title keywords, or local PDF path",
    ),
    idea: str | None = typer.Argument(
        None,
        help='Research idea about the paper (e.g. "Apply attention to drug discovery")',
    ),
    topic: str | None = typer.Option(
        None,
        "--topic",
        "-t",
        help="Opening idea (alias for positional argument)",
    ),
    idea_id: str | None = typer.Option(
        None,
        "--idea-id",
        help="Link session to an existing saved idea",
    ),
) -> None:
    """Debate a research idea grounded in a specific paper (search + load first)."""
    cfg = _ensure_api_key()
    llm = LLMClient.from_config(cfg)
    opening = idea or topic
    code = run_discuss(
        cfg,
        llm,
        console,
        paper_query=paper,
        opening_topic=opening,
        idea_id=idea_id,
    )
    raise typer.Exit(code=code)


@ideas_app.command("list")
def ideas_list() -> None:
    """List ideas grouped by status."""
    cfg = _load_config()
    code = run_ideas_list(cfg, console)
    raise typer.Exit(code=code)


@ideas_app.command("show")
def ideas_show(
    idea_id: str = typer.Argument(..., help="Idea id or prefix"),
) -> None:
    """Show idea details, score history, and linked discussions."""
    cfg = _load_config()
    code = run_ideas_show(cfg, console, idea_id)
    raise typer.Exit(code=code)


@ideas_app.command("update")
def ideas_update(
    idea_id: str = typer.Argument(..., help="Idea id or prefix"),
    status: str | None = typer.Option(
        None,
        "--status",
        help=f"New status: {', '.join(IDEA_STATUSES)}",
    ),
    feedback: str | None = typer.Option(
        None,
        "--feedback",
        help='Record score disagreement (e.g. "I think you overestimated")',
    ),
) -> None:
    """Update idea status or record score feedback without changing the score."""
    cfg = _load_config()
    if status is not None and status not in IDEA_STATUSES:
        console.print(f"[red]Error:[/red] Invalid status. Choose: {', '.join(IDEA_STATUSES)}")
        raise typer.Exit(code=1)
    new_status: IdeaStatus | None = status  # validated against IDEA_STATUSES above
    code = run_ideas_update(
        cfg,
        console,
        idea_id,
        status=new_status,
        feedback=feedback,
    )
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
