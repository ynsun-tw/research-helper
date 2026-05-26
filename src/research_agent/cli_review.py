"""CLI handler for ``research review`` (M4 S4.3.1 / S4.3.2).

Non-interactive mode: Analyst + Critic in parallel, then Scribe
revision. Interactive mode (S4.3.2): present each reviewer issue
and suggestion, let the user pick which ones to act on, then call
Scribe.revise with the filtered reviews. Persist the (original,
revised, selected_*, rejected_*) tuple so S4.1.3 can mine it for
fingerprint updates.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from research_agent.agents.orchestrator import Orchestrator
from research_agent.agents.scribe import Draft, Scribe, normalize_section
from research_agent.agents.writing_pipeline import ReviewedDraft, WritingReview
from research_agent.config import Config
from research_agent.core.llm import LLMClient
from research_agent.storage.database import Database
from research_agent.storage.draft_revisions import DraftRevision, DraftRevisionRepository
from research_agent.style.fingerprint import Fingerprint

# Type of the y/n prompt callable. Single-letter answer, case-insensitive.
PromptFn = Callable[[str], str]


@dataclass(slots=True)
class SelectionSummary:
    """Records what the user accepted vs rejected in interactive mode."""

    selected_issues: list[str]
    selected_suggestions: list[str]
    rejected_issues: list[str]
    rejected_suggestions: list[str]


@dataclass(slots=True)
class ReviewResult:
    reviewed: ReviewedDraft
    selection: SelectionSummary | None = None
    saved_revision: DraftRevision | None = None


def run_review(
    cfg: Config,
    console: Console,
    *,
    draft_path: Path | None = None,
    draft_text: str = "",
    section: str = "introduction",
    target_words: int = 0,
    output: Path | None = None,
    interactive: bool = False,
    save: bool = False,
    prompt_fn: PromptFn | None = None,
    orchestrator: Orchestrator | None = None,
    scribe: Scribe | None = None,
    revisions_repo: DraftRevisionRepository | None = None,
) -> ReviewResult:
    """Run the writing-review pipeline.

    In ``interactive`` mode we surface analyst + critic reviews one
    item at a time, ask the user [y/n] per issue and suggestion,
    then call Scribe.revise with only the selected items. A unified
    diff between original and revised is printed at the end. When
    ``save`` is true (default ``True`` in interactive mode) the
    revision pair lands in the ``draft_revisions`` SQLite table so
    S4.1.3 can learn from it.
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

    selection: SelectionSummary | None = None
    if interactive:
        raw_reviews = orchestrator.collect_writing_reviews(draft)
        _render_original(console, draft)
        for r in raw_reviews:
            console.print(_review_panel(r))
        prompt = prompt_fn or _default_prompt
        filtered, selection = _interactive_select(console, raw_reviews, prompt)
        revised = scribe.revise(draft, filtered, fingerprint=fingerprint)
        reviewed = ReviewedDraft(original=draft, reviews=raw_reviews, revised=revised)
        _render_diff(console, draft, revised)
    else:
        reviewed = orchestrator.writing_review_pipeline(
            scribe, draft, fingerprint=fingerprint
        )
        _render_reviewed(console, reviewed)

    saved: DraftRevision | None = None
    if save:
        saved = _save_revision(
            cfg,
            console,
            reviewed=reviewed,
            selection=selection,
            interactive=interactive,
            repo_override=revisions_repo,
        )

    if output is not None:
        _persist_reviewed(output, reviewed, selection=selection)
        console.print(f"[green]✓[/green] Wrote review bundle to [bold]{output}[/bold]")

    return ReviewResult(reviewed=reviewed, selection=selection, saved_revision=saved)


def _default_prompt(question: str) -> str:
    """Wraps :func:`input` so tests can supply their own iterator."""
    try:
        return input(question)
    except EOFError:
        return ""


def _interactive_select(
    console: Console,
    reviews: list[WritingReview],
    prompt: PromptFn,
) -> tuple[list[WritingReview], SelectionSummary]:
    """Walk the user through every issue / suggestion and return the
    filtered review list plus a :class:`SelectionSummary`."""
    selected_issues: list[str] = []
    selected_suggestions: list[str] = []
    rejected_issues: list[str] = []
    rejected_suggestions: list[str] = []
    filtered: list[WritingReview] = []

    console.print(
        "[bold]Pick which suggestions to act on[/bold]  "
        "[dim](y = accept, anything else = reject, Enter = reject; Ctrl-D ends)[/dim]"
    )
    for r in reviews:
        kept_issues: list[str] = []
        kept_suggestions: list[str] = []
        for item in r.issues:
            ans = prompt(f"[{r.role}] issue → {item}  [y/N] ").strip().lower()
            if ans == "y":
                kept_issues.append(item)
                selected_issues.append(item)
            else:
                rejected_issues.append(item)
        for item in r.suggestions:
            ans = prompt(f"[{r.role}] suggestion → {item}  [y/N] ").strip().lower()
            if ans == "y":
                kept_suggestions.append(item)
                selected_suggestions.append(item)
            else:
                rejected_suggestions.append(item)
        if kept_issues or kept_suggestions:
            filtered.append(
                WritingReview(
                    role=r.role,
                    issues=kept_issues,
                    suggestions=kept_suggestions,
                    summary=r.summary,
                    raw_response=r.raw_response,
                )
            )

    return filtered, SelectionSummary(
        selected_issues=selected_issues,
        selected_suggestions=selected_suggestions,
        rejected_issues=rejected_issues,
        rejected_suggestions=rejected_suggestions,
    )


def _save_revision(
    cfg: Config,
    console: Console,
    *,
    reviewed: ReviewedDraft,
    selection: SelectionSummary | None,
    interactive: bool,
    repo_override: DraftRevisionRepository | None = None,
) -> DraftRevision | None:
    """Persist the (original, revised) pair to ``draft_revisions``.

    No-op when the revision is byte-identical to the original (Scribe
    short-circuited with no actionable feedback).
    """
    if reviewed.revised.text.strip() == reviewed.original.text.strip():
        return None
    repo = repo_override
    db_owned: Database | None = None
    if repo is None:
        try:
            db_owned = Database(cfg.db_path)
            repo = DraftRevisionRepository(db_owned)
        except Exception as exc:
            console.print(
                f"[yellow]Could not persist revision pair:[/yellow] {exc}"
            )
            return None
    try:
        rev = repo.add(
            section=reviewed.original.section,
            original_text=reviewed.original.text,
            revised_text=reviewed.revised.text,
            selected_issues=(selection.selected_issues if selection else None),
            selected_suggestions=(
                selection.selected_suggestions if selection else None
            ),
            rejected_issues=(selection.rejected_issues if selection else None),
            rejected_suggestions=(
                selection.rejected_suggestions if selection else None
            ),
            interactive=interactive,
        )
    finally:
        if db_owned is not None:
            db_owned.close()
    console.print(
        f"[green]✓[/green] Saved revision pair [dim]({rev.id[:8]})[/dim] for "
        f"future style learning."
    )
    return rev


def _render_original(console: Console, draft: Draft) -> None:
    console.print(
        Panel(
            draft.text or "[dim](empty)[/dim]",
            title=(
                f"[bold]Original draft[/bold] · {draft.section} · "
                f"{draft.word_count} words"
            ),
            border_style="dim",
        )
    )


def _render_diff(console: Console, original: Draft, revised: Draft) -> None:
    diff_lines = list(
        difflib.unified_diff(
            original.text.splitlines(),
            revised.text.splitlines(),
            fromfile="original",
            tofile="revised",
            lineterm="",
        )
    )
    if not diff_lines:
        console.print(
            Panel(
                "[dim]No textual change between original and revised draft.[/dim]",
                title="[bold]Diff[/bold]",
                border_style="dim",
            )
        )
        return
    body = Text()
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            body.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("---"):
            body.append(line + "\n", style="red")
        elif line.startswith("@@"):
            body.append(line + "\n", style="cyan")
        else:
            body.append(line + "\n", style="dim")
    console.print(Panel(body, title="[bold]Diff[/bold]", border_style="green"))


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


def _persist_reviewed(
    path: Path,
    reviewed: ReviewedDraft,
    *,
    selection: SelectionSummary | None = None,
) -> None:
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
    if selection is not None:
        lines.append("## User selection")
        lines.append("")
        if selection.selected_issues:
            lines.append("**Accepted issues**")
            lines.extend(f"- {x}" for x in selection.selected_issues)
            lines.append("")
        if selection.selected_suggestions:
            lines.append("**Accepted suggestions**")
            lines.extend(f"- {x}" for x in selection.selected_suggestions)
            lines.append("")
        if selection.rejected_issues or selection.rejected_suggestions:
            lines.append("**Rejected**")
            lines.extend(f"- (issue) {x}" for x in selection.rejected_issues)
            lines.extend(
                f"- (suggestion) {x}" for x in selection.rejected_suggestions
            )
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
