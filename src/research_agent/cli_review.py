"""CLI handler for ``research review`` (M4 S4.3.1).

Takes an existing draft file (the user's own writing, or a Scribe
output saved earlier), runs the auto-review pipeline (Analyst +
Critic in parallel, then Scribe revision), and renders the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from research_agent.agents.orchestrator import Orchestrator
from research_agent.agents.scribe import Draft, Scribe, normalize_section
from research_agent.agents.writing_pipeline import ReviewedDraft, WritingReview
from research_agent.config import Config
from research_agent.core.llm import LLMClient
from research_agent.style.fingerprint import Fingerprint


@dataclass(slots=True)
class ReviewResult:
    reviewed: ReviewedDraft


def run_review(
    cfg: Config,
    console: Console,
    *,
    draft_path: Path | None = None,
    draft_text: str = "",
    section: str = "introduction",
    target_words: int = 0,
    output: Path | None = None,
    orchestrator: Orchestrator | None = None,
    scribe: Scribe | None = None,
) -> ReviewResult:
    """Run the Scribe → Analyst+Critic → Scribe pipeline.

    Either ``draft_path`` or ``draft_text`` must be provided.
    ``orchestrator`` / ``scribe`` are injectable for tests.
    """
    if draft_path is not None and not draft_text:
        draft_text = draft_path.read_text(encoding="utf-8")
    if not draft_text.strip():
        raise ValueError("draft is empty; pass a path with content or --text.")
    section_norm = normalize_section(section)
    word_count = len(draft_text.split())
    if target_words <= 0:
        target_words = word_count

    draft = Draft(
        section=section_norm,
        version="A",
        variant_label="user input",
        text=draft_text.strip(),
        style_note="",
        word_count=word_count,
        target_words=target_words,
    )

    fingerprint: Fingerprint | None = None
    if cfg.fingerprint_path.exists():
        try:
            fingerprint = Fingerprint.load_from(cfg.fingerprint_path)
        except Exception as exc:
            console.print(
                f"[yellow]Could not load fingerprint:[/yellow] {exc}. "
                "Revision will use generic academic prose."
            )

    if orchestrator is None or scribe is None:
        llm = LLMClient.from_config(cfg)
        if orchestrator is None:
            orchestrator = Orchestrator(llm, language=cfg.language)
        if scribe is None:
            scribe = Scribe(llm, language=cfg.language)

    reviewed = orchestrator.writing_review_pipeline(
        scribe, draft, fingerprint=fingerprint
    )

    _render_reviewed(console, reviewed)
    if output is not None:
        _persist_reviewed(output, reviewed)
        console.print(f"[green]✓[/green] Wrote review bundle to [bold]{output}[/bold]")

    return ReviewResult(reviewed=reviewed)


def _render_reviewed(console: Console, reviewed: ReviewedDraft) -> None:
    section = reviewed.original.section
    console.print(
        Panel(
            reviewed.original.text or "[dim](empty)[/dim]",
            title=(
                f"[bold]Original draft[/bold] · {section} · "
                f"{reviewed.original.word_count} words"
            ),
            border_style="dim",
        )
    )
    for review in reviewed.reviews:
        console.print(_review_panel(review))
    console.print(
        Panel(
            reviewed.revised.text or "[dim](empty)[/dim]",
            title=(
                f"[bold]Revised draft[/bold] · {reviewed.revised.style_note}"
                if reviewed.revised.style_note
                else f"[bold]Revised draft[/bold] · {section}"
            ),
            border_style="green",
        )
    )


def _review_panel(review: WritingReview) -> Panel:
    body_lines: list[str] = []
    if review.summary:
        body_lines.append(f"[italic dim]{review.summary}[/italic dim]")
        body_lines.append("")
    if review.issues:
        body_lines.append("[bold]Issues[/bold]")
        body_lines.extend(f"- {x}" for x in review.issues)
    else:
        body_lines.append("[dim]No issues flagged.[/dim]")
    if review.suggestions:
        body_lines.append("")
        body_lines.append("[bold]Suggestions[/bold]")
        body_lines.extend(f"- {x}" for x in review.suggestions)
    border = "yellow" if review.role == "analyst" else "red"
    return Panel(
        "\n".join(body_lines),
        title=f"[bold]{review.role.title()} review[/bold]",
        border_style=border,
    )


def _persist_reviewed(path: Path, reviewed: ReviewedDraft) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Review — {reviewed.original.section}",
        "",
        "## Original",
        "",
        reviewed.original.text or "",
        "",
    ]
    for r in reviewed.reviews:
        lines.append(f"## {r.role.title()} review")
        if r.summary:
            lines.append(f"_{r.summary}_")
            lines.append("")
        if r.issues:
            lines.append("**Issues**")
            lines.extend(f"- {x}" for x in r.issues)
            lines.append("")
        if r.suggestions:
            lines.append("**Suggestions**")
            lines.extend(f"- {x}" for x in r.suggestions)
            lines.append("")
    lines.extend(
        [
            "## Revised",
            "",
            f"_{reviewed.revised.style_note}_" if reviewed.revised.style_note else "",
            "",
            reviewed.revised.text or "",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
