"""CLI handler for ``research check`` — self-plagiarism scan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from research_agent.config import Config
from research_agent.storage.database import Database
from research_agent.style.plagiarism import (
    DEFAULT_THRESHOLD,
    PlagiarismDetector,
    SimilarityReport,
    split_into_paragraphs,
)
from research_agent.style.samples import StyleSampleRepository


@dataclass(slots=True)
class CheckResult:
    report: SimilarityReport


def run_check(
    cfg: Config,
    console: Console,
    *,
    draft_path: Path | None = None,
    draft_text: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    output: Path | None = None,
    samples_override: list[tuple[str, str]] | None = None,
) -> CheckResult:
    """Compare a draft against the user's ``style_samples`` corpus.

    ``samples_override`` lets tests skip the SQLite round-trip; in
    production the corpus is loaded from ``cfg.db_path``.
    """
    if draft_path is not None and not draft_text:
        draft_text = draft_path.read_text(encoding="utf-8")
    if not draft_text.strip():
        raise ValueError("draft is empty; pass a path with content or --text.")

    if samples_override is None:
        db = Database(cfg.db_path)
        try:
            repo = StyleSampleRepository(db)
            corpus = [(s.paper_id, s.paragraph) for s in repo.iter_all()]
        finally:
            db.close()
    else:
        corpus = list(samples_override)

    if not corpus:
        console.print(
            "[yellow]No style samples to compare against.[/yellow] "
            "Train first with [bold]research style train[/bold]."
        )
        report = SimilarityReport(
            threshold=threshold,
            draft_paragraph_count=len(split_into_paragraphs(draft_text)),
        )
        return CheckResult(report=report)

    paragraphs = split_into_paragraphs(draft_text)
    detector = PlagiarismDetector(threshold=threshold)
    report = detector.check(paragraphs, corpus)
    _render_report(console, report)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_markdown(), encoding="utf-8")
        console.print(
            f"[green]✓[/green] Wrote similarity report to [bold]{output}[/bold]"
        )
    return CheckResult(report=report)


def _render_report(console: Console, report: SimilarityReport) -> None:
    header = (
        f"[bold]Self-plagiarism check[/bold]  "
        f"[dim](threshold {report.threshold:.0%}, "
        f"{report.draft_paragraph_count} draft paragraphs scanned)[/dim]"
    )
    if report.is_clean:
        console.print(
            Panel(
                "[green]No matches above threshold. You're good to ship.[/green]",
                title=header,
                border_style="green",
            )
        )
        return
    console.print(
        Panel(
            f"[red]{len(report.matches)} match(es) at or above "
            f"{report.threshold:.0%} similarity.[/red]",
            title=header,
            border_style="red",
        )
    )
    console.print(Markdown(report.to_markdown()))
