"""CLI handlers for ``research style`` commands.

These are kept out of :mod:`research_agent.cli` so the actual Typer
file stays scannable. The functions also have stable signatures so
the chat layer and the test suite can drive them without going
through Typer's argument parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from research_agent.config import Config
from research_agent.core.loader import PaperLoadError, load_paper
from research_agent.storage.database import Database
from research_agent.style.extractor import extract_samples
from research_agent.style.samples import StyleSampleRepository


@dataclass(slots=True)
class StyleTrainResult:
    """Tally returned by :func:`run_style_train`. Useful for tests + chat."""

    sources_processed: int
    sources_failed: int
    paragraphs_added: int
    paragraphs_skipped: int
    per_source: list[tuple[str, int, str]]  # (label, paragraph_count, status)


def _collect_pdfs_in_dir(directory: Path) -> list[Path]:
    """Return sorted .pdf files under ``directory`` (non-recursive).

    Style training is meant to be intentional - the user puts a handful
    of representative papers in one folder. Recursing into nested dirs
    surprises people who use deep archives, so we don't.
    """
    if not directory.exists():
        raise PaperLoadError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise PaperLoadError(f"Not a directory: {directory}")
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def run_style_train(
    cfg: Config,
    console: Console,
    *,
    sources: list[str] | None = None,
    directory: Path | None = None,
    replace: bool = True,
) -> StyleTrainResult:
    """Import style samples from one or more sources.

    ``sources`` can mix ``arxiv:<id>`` shorthands and direct paths to
    local PDFs. ``directory`` enumerates every PDF in one folder.
    When ``replace`` is true (default) we delete prior samples for any
    paper id we re-import - re-training on the same paper should not
    duplicate data.
    """
    sources = sources or []
    db = Database(cfg.db_path)
    try:
        repo = StyleSampleRepository(db)

        per_source: list[tuple[str, int, str]] = []
        paragraphs_added = 0
        paragraphs_skipped = 0
        ok = 0
        failed = 0

        all_sources: list[tuple[str, str]] = []  # (label, source)
        for src in sources:
            all_sources.append((src, src))
        if directory is not None:
            try:
                for pdf in _collect_pdfs_in_dir(directory):
                    all_sources.append((pdf.name, str(pdf)))
            except PaperLoadError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                return StyleTrainResult(0, 0, 0, 0, [])

        if not all_sources:
            console.print(
                "[yellow]No sources provided.[/yellow] "
                "Pass arXiv ids/PDF paths, or --dir <folder>."
            )
            return StyleTrainResult(0, 0, 0, 0, [])

        for label, src in all_sources:
            try:
                paper = load_paper(src, cache_dir=cfg.pdf_cache_dir)
            except Exception as exc:
                # We want a single bad PDF to not abort the whole training
                # run - the user expects "process every file in the folder".
                console.print(f"[red]✗[/red] {label}: {exc}")
                per_source.append((label, 0, f"error: {exc}"))
                failed += 1
                continue

            samples = extract_samples(paper)
            if not samples:
                console.print(
                    f"[yellow]~[/yellow] {label}: no usable paragraphs after filtering"
                )
                per_source.append((label, 0, "no usable paragraphs"))
                ok += 1  # parse succeeded; filter just rejected everything
                paragraphs_skipped += len(extract_samples(paper))  # always 0 here
                continue

            if replace:
                removed = repo.delete_for_paper(paper.id)
                if removed:
                    console.print(
                        f"[dim]· {label}: replaced {removed} prior sample(s)[/dim]"
                    )
            added = repo.bulk_add(samples)
            paragraphs_added += added
            per_source.append((label, added, "ok"))
            ok += 1
            console.print(f"[green]✓[/green] {label}: {added} paragraphs")

        # Render summary table
        table = Table(title="Style training summary", show_header=True)
        table.add_column("Source")
        table.add_column("Paragraphs", justify="right")
        table.add_column("Status")
        for label, count, status in per_source:
            table.add_row(label, str(count), status)
        console.print(table)
        console.print(
            f"[bold]{paragraphs_added}[/bold] paragraphs across "
            f"{ok}/{ok + failed} source(s) → ~/.research-agent/memory.db"
        )
        return StyleTrainResult(
            sources_processed=ok,
            sources_failed=failed,
            paragraphs_added=paragraphs_added,
            paragraphs_skipped=paragraphs_skipped,
            per_source=per_source,
        )
    finally:
        db.close()


def run_style_show(cfg: Config, console: Console) -> int:
    """Print a short summary of the current style sample corpus."""
    db = Database(cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        total = repo.count()
        by_paper = repo.count_by_paper()
        if total == 0:
            console.print(
                "[yellow]No style samples yet.[/yellow] "
                "Run [bold]research style train[/bold] first."
            )
            return 0
        table = Table(title="Style corpus", show_header=True)
        table.add_column("Source paper")
        table.add_column("Paragraphs", justify="right")
        for paper_id, count in sorted(by_paper.items()):
            table.add_row(paper_id, str(count))
        console.print(table)
        console.print(f"[bold]{total}[/bold] paragraphs across {len(by_paper)} paper(s)")
        return 0
    finally:
        db.close()
