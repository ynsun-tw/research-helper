"""Rich terminal formatting for CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from research_agent.agents.base import AgentResponse
from research_agent.agents.debate import DebateResult, FollowUpResult
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


def render_debate_result(console: Console, result: DebateResult) -> None:
    console.print(
        Panel(
            _debate_section("Supports", result.supports, empty="(none yet)"),
            title=f"[bold green]📄 Analyst[/bold green] · confidence {result.confidence:.0%}",
            border_style="green",
        )
    )
    console.print(
        Panel(
            _debate_section("Objections", result.objections, empty="(none)"),
            title=f"[bold red]🔴 Critic[/bold red] · score {result.score:.0f}/9",
            border_style="red",
        )
    )
    if result.score_reason:
        console.print(f"[dim]Score reason:[/dim] {result.score_reason}")
    if result.suggestions:
        console.print(
            Panel(
                _debate_section("Suggestions", result.suggestions),
                title="[bold yellow]💡 Suggestions[/bold yellow]",
                border_style="yellow",
            )
        )
    if result.score_delta is not None:
        sign = "+" if result.score_delta >= 0 else ""
        console.print(f"[dim]Score change vs last round:[/dim] {sign}{result.score_delta:.0f}")


def _debate_section(title: str, items: list[str], *, empty: str = "(none)") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def render_followup_turn(console: Console, followup: FollowUpResult) -> None:
    if followup.analyst_conclusion:
        console.print(
            Panel(
                Markdown(followup.analyst_conclusion),
                title="[bold green]📄 Analyst[/bold green]",
                border_style="green",
            )
        )
    if followup.critic_conclusion:
        console.print(
            Panel(
                Markdown(followup.critic_conclusion),
                title="[bold red]🔴 Critic[/bold red]",
                border_style="red",
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
