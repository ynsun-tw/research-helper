"""Rich terminal formatting for CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from research_agent.agents.base import AgentResponse
from research_agent.agents.orchestrator import AggregatedAnalysis
from research_agent.core.paper import Paper


def render_paper_header(console: Console, paper: Paper) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold]Title[/bold]", paper.title)
    if paper.authors:
        table.add_row("[bold]Authors[/bold]", ", ".join(paper.authors))
    table.add_row("[bold]ID[/bold]", paper.id)
    console.print(Panel(table, title="[bold cyan]Paper[/bold cyan]", border_style="cyan"))


def render_read_report(console: Console, paper: Paper, report: AggregatedAnalysis) -> None:
    render_paper_header(console, paper)

    console.print(
        Panel(
            Markdown(report.analyst.to_agent_response().content),
            title="[bold green]📄 Analyst[/bold green]",
            border_style="green",
        )
    )
    console.print(
        Panel(
            Markdown(report.critic.to_agent_response().content),
            title="[bold red]🔴 Critic[/bold red]",
            border_style="red",
        )
    )

    synthesis = _synthesis_markdown(report)
    console.print(
        Panel(
            Markdown(synthesis),
            title="[bold blue]📊 Synthesis[/bold blue]",
            border_style="blue",
        )
    )


def render_discuss_turn(
    console: Console,
    analyst: AgentResponse,
    critic: AgentResponse,
) -> None:
    console.print(
        Panel(
            Markdown(analyst.content),
            title="[bold green]📄 Analyst[/bold green]",
            border_style="green",
        )
    )
    console.print(
        Panel(
            Markdown(critic.content),
            title="[bold red]🔴 Critic[/bold red]",
            border_style="red",
        )
    )


def _synthesis_markdown(report: AggregatedAnalysis) -> str:
    lines = [report.summary, ""]
    if report.consensus:
        lines.append("### Consensus")
        lines.extend(f"- {c}" for c in report.consensus)
        lines.append("")
    if report.conflicts:
        lines.append("### Conflicts / tensions")
        lines.extend(f"- {c}" for c in report.conflicts)
    return "\n".join(lines)
