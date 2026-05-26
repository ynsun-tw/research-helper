"""CLI handlers for ``research ideas`` commands."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from research_agent.config import Config
from research_agent.core.idea import IDEA_STATUSES, IdeaStatus, InvalidStatusTransition
from research_agent.storage.database import Database
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository


def run_ideas_list(cfg: Config, console: Console) -> int:
    db = Database(cfg.db_path)
    try:
        repo = IdeaRepository(db)
        grouped = repo.list_by_status()
        if not any(grouped.values()):
            console.print("[yellow]No ideas saved yet.[/yellow]")
            return 0
        for status in IDEA_STATUSES:
            ideas = grouped.get(status, [])
            if not ideas:
                continue
            table = Table(title=f"Status: {status}", show_header=True)
            table.add_column("ID", style="dim", max_width=12)
            table.add_column("Title")
            table.add_column("Score", justify="right")
            for idea in ideas:
                score = f"{idea.critic_score:.0f}/9" if idea.critic_score is not None else "—"
                table.add_row(idea.id[:8] + "…", idea.title, score)
            console.print(table)
        return 0
    finally:
        db.close()


def run_ideas_show(cfg: Config, console: Console, idea_id: str) -> int:
    db = Database(cfg.db_path)
    try:
        repo = IdeaRepository(db)
        discussions = DiscussionRepository(db)
        idea = repo.get(idea_id)
        if idea is None:
            for candidate in repo.list_all():
                if candidate.id.startswith(idea_id):
                    idea = candidate
                    break
        if idea is None:
            console.print(f"[red]Error:[/red] Idea not found: {idea_id}")
            return 1

        console.print(f"[bold]{idea.title}[/bold]  [dim]({idea.id})[/dim]")
        console.print(f"Status: [cyan]{idea.status}[/cyan]")
        if idea.description:
            console.print(f"\n{idea.description}\n")
        if idea.critic_objections:
            console.print("[bold]Latest objections[/bold]")
            for obj in idea.critic_objections:
                console.print(f"  - {obj}")
        if idea.score_history:
            console.print("\n[bold]Score history[/bold]")
            for entry in idea.score_history:
                sid = f" session={entry.session_id[:8]}…" if entry.session_id else ""
                console.print(f"  - {entry.score:.0f}/9 — {entry.reason}{sid}")
        if idea.activation_conditions:
            console.print(
                "\n[bold]Activation conditions[/bold] "
                "(matched against future search hits)"
            )
            for cond in idea.activation_conditions:
                console.print(f"  - {cond}")
        if idea.user_score_feedback:
            console.print("\n[bold]Your score feedback[/bold] (recorded, does not change scores)")
            for note in idea.user_score_feedback:
                console.print(f"  - {note}")

        sessions = repo.session_ids(idea.id)
        if sessions:
            console.print("\n[bold]Discussion sessions[/bold]")
            for sid in sessions:
                msgs = discussions.list_session(sid)
                console.print(f"  - {sid[:8]}… ({len(msgs)} message(s))")
        return 0
    finally:
        db.close()


def run_ideas_update(
    cfg: Config,
    console: Console,
    idea_id: str,
    *,
    status: IdeaStatus | None = None,
    feedback: str | None = None,
    conditions: list[str] | None = None,
    clear_conditions: bool = False,
) -> int:
    db = Database(cfg.db_path)
    try:
        repo = IdeaRepository(db)
        resolved = _resolve_idea_id(repo, idea_id)
        if resolved is None:
            console.print(f"[red]Error:[/red] Idea not found: {idea_id}")
            return 1
        if status is not None:
            try:
                repo.update_status(resolved, status)
                console.print(f"[green]✓[/green] Status updated to [bold]{status}[/bold]")
            except InvalidStatusTransition as exc:
                console.print(f"[red]Error:[/red] {exc}")
                return 1
        if feedback:
            repo.add_user_score_feedback(resolved, feedback)
            console.print("[green]✓[/green] Score feedback recorded (current score unchanged)")
        if clear_conditions:
            repo.clear_activation_conditions(resolved)
            console.print("[green]✓[/green] Activation conditions cleared")
        for condition in conditions or []:
            cleaned = condition.strip()
            if not cleaned:
                continue
            repo.add_activation_condition(resolved, cleaned)
            console.print(
                f"[green]✓[/green] Activation condition added: [italic]{cleaned}[/italic]"
            )
        return 0
    finally:
        db.close()


def _resolve_idea_id(repo: IdeaRepository, idea_id: str) -> str | None:
    if repo.get(idea_id) is not None:
        return idea_id
    for candidate in repo.list_all():
        if candidate.id.startswith(idea_id):
            return candidate.id
    return None
