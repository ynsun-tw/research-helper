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

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from research_agent.agents.meta_memory import MetaMemory
from research_agent.chat import run_chat
from research_agent.cli_review import run_review
from research_agent.cli_style import (
    run_style_fingerprint,
    run_style_show,
    run_style_train,
)
from research_agent.cli_write import run_write
from research_agent.config import Config, ConfigError
from research_agent.core.language import language_label
from research_agent.core.llm import LLMClient
from research_agent.storage.database import Database

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
style_app = typer.Typer(
    help="Train Scribe on your writing style (import samples, build fingerprints)."
)
app.add_typer(style_app, name="style")

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


@app.command("insights")
def insights(
    since: str = typer.Option(
        "",
        "--since",
        help=(
            "Limit to recent activity. Accepts: 7d, 30d, 6m, 1y, all (default)."
        ),
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Optional path to write the Markdown report. Prints to stdout otherwise.",
    ),
) -> None:
    """Generate a research-activity Markdown summary from local storage.

    Rolls up papers (top tags, authors, venues), ideas (by status,
    most-debated, top-scored), and discussion activity. The report is
    deterministic - no LLM call - so it's safe to run anywhere.
    """
    cfg = _load_config()
    since_days: int | None = None
    if since.strip():
        s = since.strip().lower()
        unit_map = {"d": 1, "w": 7, "m": 30, "y": 365}
        try:
            if s in ("all", "alltime", "all-time"):
                since_days = None
            elif s[-1] in unit_map and s[:-1].isdigit():
                since_days = int(s[:-1]) * unit_map[s[-1]]
            elif s.isdigit():
                since_days = int(s)
            else:
                raise ValueError(f"Unrecognised --since value: {since!r}")
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    db = Database(cfg.db_path)
    try:
        report = MetaMemory(db).compute(since_days=since_days)
    finally:
        db.close()
    markdown = report.to_markdown()
    if output.strip():
        out_path = Path(output).expanduser()
        out_path.write_text(markdown, encoding="utf-8")
        console.print(f"[green]✓[/green] Wrote insights to [bold]{out_path}[/bold]")
        return
    from rich.markdown import Markdown

    console.print(Markdown(markdown))


@style_app.command("train")
def style_train(
    sources: list[str] = typer.Argument(
        None,
        help=(
            "Sources to learn from. Each item is either 'arxiv:<id>' "
            "(or a bare arXiv id) or a path to a local PDF. Combine "
            "with --dir to also pull every PDF from a folder."
        ),
    ),
    directory: str = typer.Option(
        "",
        "--dir",
        help="Optional folder to scan for .pdf files (non-recursive).",
    ),
    append: bool = typer.Option(
        False,
        "--append",
        help=(
            "Keep prior samples for any paper re-imported in this run. "
            "By default each source's old samples are replaced."
        ),
    ),
) -> None:
    """Import prose paragraphs from your own papers for style training.

    Examples::

        research style train arxiv:2301.07041 arxiv:2305.14314
        research style train --dir ~/papers
        research style train ~/papers/my-thesis.pdf
    """
    cfg = _load_config()
    src_list = [s for s in (sources or []) if s and s.strip()]
    dir_path = None
    if directory.strip():
        dir_path = Path(directory).expanduser()

    result = run_style_train(
        cfg,
        console,
        sources=src_list,
        directory=dir_path,
        replace=not append,
    )
    if result.sources_failed and not result.sources_processed:
        raise typer.Exit(code=1)


@style_app.command("show")
def style_show() -> None:
    """Print a summary of the current style training corpus."""
    cfg = _load_config()
    code = run_style_show(cfg, console)
    raise typer.Exit(code=code)


@app.command("write")
def write_command(
    section: str = typer.Argument(
        ...,
        help=(
            "Section to draft: abstract, introduction, related_work, "
            "method, results, discussion, conclusion (aliases like "
            "'intro' / 'methods' / 'experiments' are accepted)."
        ),
    ),
    context: str = typer.Option(
        "",
        "--context",
        help=(
            "Free-form research context for the Scribe (e.g. 'this paper "
            "studies sparse top-k attention for 32k context'). Used to "
            "ground the draft."
        ),
    ),
    target_words: int = typer.Option(
        300,
        "--words",
        help="Target word count per draft (±20% tolerance).",
    ),
    versions: int = typer.Option(
        3,
        "--versions",
        "-n",
        help="Number of stylistic variants to generate.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Optional Markdown file to persist all drafts to.",
    ),
    check_against: list[str] = typer.Option(
        None,
        "--check-against",
        help=(
            "Path to an existing draft file the Scribe should stay "
            "consistent with (no duplication, no contradiction). "
            "Repeatable - pass once per file."
        ),
    ),
    sequential: bool = typer.Option(
        False,
        "--sequential",
        help="Issue LLM calls one at a time (default: parallel via thread pool).",
    ),
) -> None:
    """Have Scribe draft a section in your voice.

    Builds N variants (default 3) using the fingerprint at
    ``~/.research-agent/style/fingerprint.json``. If no fingerprint is
    available the Scribe falls back to generic academic prose and
    says so in each draft's style note.

    With ``--context`` the Scribe also auto-pulls related ideas and
    recent discussion excerpts from your memory store. With
    ``--check-against`` it sees the body of an existing draft and is
    told not to duplicate or contradict it.
    """
    cfg = _ensure_api_key()
    out_path = Path(output).expanduser() if output.strip() else None
    check_paths = [Path(p).expanduser() for p in (check_against or []) if p and p.strip()]
    run_write(
        cfg,
        console,
        section=section,
        context=context,
        check_against=check_paths,
        target_words=target_words,
        versions=versions,
        output=out_path,
        parallel=not sequential,
    )


@app.command("review")
def review_command(
    draft_file: str = typer.Argument(
        ...,
        help="Path to a Markdown / text file containing the draft to review.",
    ),
    section: str = typer.Option(
        "introduction",
        "--section",
        help=(
            "Which section the draft is from (abstract, introduction, "
            "related_work, method, results, discussion, conclusion). "
            "Drives the reviewer prompts."
        ),
    ),
    target_words: int = typer.Option(
        0,
        "--words",
        help=(
            "Target word count the revised draft should aim for. "
            "Defaults to the original draft's length."
        ),
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Optional Markdown file to persist the review bundle to.",
    ),
) -> None:
    """Run the auto-review pipeline: Scribe → Analyst+Critic → Scribe.

    Analyst flags weak argumentation and missing differentiation from
    related work; Critic flags overclaim and unsupported conclusions;
    Scribe produces a revised draft that addresses both reviews.
    """
    cfg = _ensure_api_key()
    path = Path(draft_file).expanduser()
    if not path.exists():
        console.print(f"[red]Error:[/red] {path} does not exist.")
        raise typer.Exit(code=1)
    out_path = Path(output).expanduser() if output.strip() else None
    run_review(
        cfg,
        console,
        draft_path=path,
        section=section,
        target_words=target_words,
        output=out_path,
    )


@style_app.command("fingerprint")
def style_fingerprint() -> None:
    """Build a style fingerprint from the imported samples.

    Aggregates macro structure (section openers, related-work
    organization), micro statistics (sentence-length distribution,
    transition / hedging / confidence rates, type-token ratio), and
    personal markers (citation format, figure ref style, em-dash
    rate). Writes ``~/.research-agent/style/fingerprint.json``.
    """
    cfg = _load_config()
    code = run_style_fingerprint(cfg, console)
    raise typer.Exit(code=code)


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    main()
