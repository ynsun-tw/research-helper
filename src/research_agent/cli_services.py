"""CLI command implementations (testable without Typer runner)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from research_agent.agents.analyst import AnalysisResult
from research_agent.agents.critic import CritiqueResult
from research_agent.agents.orchestrator import AggregatedAnalysis, Orchestrator
from research_agent.config import Config
from research_agent.core.llm import LLMError, LLMProvider
from research_agent.core.loader import PaperLoadError, load_paper
from research_agent.core.paper import Paper
from research_agent.storage.database import Database, PaperRepository
from research_agent.storage.discussions import DiscussionRepository
from research_agent.ui.formatting import render_discuss_turn, render_read_report


def run_read(
    source: str,
    cfg: Config,
    llm: LLMProvider,
    console: Console,
) -> int:
    """Load a paper, analyze with Analyst + Critic, render and persist."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            load_task = progress.add_task("Loading paper…", total=None)
            try:
                paper = load_paper(source, cache_dir=cfg.pdf_cache_dir)
            except PaperLoadError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                return 1
            progress.update(load_task, description="[green]✓[/green] Paper loaded")

            parse_task = progress.add_task("Parsing PDF…", total=None)
            progress.update(parse_task, description="[green]✓[/green] PDF parsed")

            analyze_task = progress.add_task("Analyzing (Analyst + Critic)…", total=None)
            orch = Orchestrator(llm)
            report = asyncio.run(orch.analyze_paper_parallel(paper))
            progress.update(analyze_task, description="[green]✓[/green] Analysis complete")

            save_task = progress.add_task("Saving results…", total=None)
            _persist_read_results(cfg, paper, report)
            progress.update(save_task, description="[green]✓[/green] Results saved")

        render_read_report(console, paper, report)
        return 0
    except LLMError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    except ValueError as exc:
        console.print(f"[red]Error:[/red] Could not parse model output: {exc}")
        return 1


def run_discuss(
    cfg: Config,
    llm: LLMProvider,
    console: Console,
    *,
    opening_topic: str | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    """Interactive REPL with dual-agent replies; persists on exit."""
    if input_fn is None:
        input_fn = console.input
    session_id = str(uuid.uuid4())
    history: list[tuple[str, str]] = []
    discussions = DiscussionRepository(Database(cfg.db_path))
    orch = Orchestrator(llm)

    console.print(
        Panel(
            "Critical discussion mode. Each reply includes Analyst + Critic views.\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to leave (session is saved).",
            title="[bold]Research Discuss[/bold]",
            border_style="magenta",
        )
    )

    try:
        if opening_topic:
            code = _handle_discuss_turn(
                opening_topic,
                history,
                discussions,
                session_id,
                orch,
                console,
            )
            if code != 0:
                return code

        while True:
            try:
                user_input = input_fn("[bold cyan]You>[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Interrupted — saving session…[/yellow]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            code = _handle_discuss_turn(
                user_input,
                history,
                discussions,
                session_id,
                orch,
                console,
            )
            if code != 0:
                return code

    finally:
        discussions.db.close()
        n = len([h for h in history if h[0] == "user"])
        console.print(f"[green]✓[/green] Session saved ({n} turn(s), id={session_id[:8]}…)")
    return 0


def _handle_discuss_turn(
    user_input: str,
    history: list[tuple[str, str]],
    discussions: DiscussionRepository,
    session_id: str,
    orch: Orchestrator,
    console: Console,
) -> int:
    history.append(("user", user_input))
    discussions.append(session_id, "user", user_input)
    try:
        with console.status("[bold]Thinking…[/bold] (Analyst + Critic)"):
            analyst, critic = asyncio.run(
                orch.discuss_turn_async(user_input, history[:-1])
            )
    except LLMError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        history.pop()
        return 0
    except (ValueError, KeyError, TypeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        history.pop()
        return 0

    render_discuss_turn(console, analyst, critic)
    history.append(("analyst", analyst.content))
    history.append(("critic", critic.content))
    discussions.append(session_id, "analyst", analyst.content)
    discussions.append(session_id, "critic", critic.content)
    return 0


def _persist_read_results(cfg: Config, paper: Paper, report: AggregatedAnalysis) -> None:
    db = Database(cfg.db_path)
    try:
        repo = PaperRepository(db)
        repo.save(paper)
        repo.save_analysis_notes(
            paper.id,
            analyst_notes=_analysis_to_dict(report.analyst),
            critic_notes=_critique_to_dict(report.critic),
        )
    finally:
        db.close()


def _analysis_to_dict(result: AnalysisResult) -> dict[str, object]:
    data = asdict(result)
    data["claimed_vs_evidence"] = [
        {"claim": p.claim, "evidence": p.evidence} for p in result.claimed_vs_evidence
    ]
    return data


def _critique_to_dict(result: CritiqueResult) -> dict[str, object]:
    return asdict(result)
