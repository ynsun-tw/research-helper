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
from research_agent.storage.draft_revisions import DraftRevisionRepository
from research_agent.style.analyzer import StyleAnalyzer
from research_agent.style.extractor import extract_samples, extract_samples_from_text
from research_agent.style.fingerprint import Fingerprint
from research_agent.style.samples import StyleSampleRepository


@dataclass
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
    finally:
        db.close()

    fp_path = cfg.fingerprint_path
    if fp_path.exists():
        try:
            fp = Fingerprint.load_from(fp_path)
        except Exception as exc:
            console.print(
                f"[yellow]Fingerprint file present but unreadable:[/yellow] {exc}"
            )
            return 0
        _render_fingerprint(console, fp, fp_path)
    else:
        console.print(
            "[dim]No fingerprint yet. Run [bold]research style fingerprint[/bold] "
            "after training.[/dim]"
        )
    return 0


def run_style_fingerprint(cfg: Config, console: Console) -> int:
    """Compute a :class:`Fingerprint` from the current corpus and persist it."""
    db = Database(cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        samples = repo.list_all()
    finally:
        db.close()
    if not samples:
        console.print(
            "[yellow]No style samples found.[/yellow] "
            "Run [bold]research style train[/bold] first."
        )
        return 1
    fp = StyleAnalyzer().analyze(samples)
    fp.save_to(cfg.fingerprint_path)
    console.print(
        f"[green]✓[/green] Fingerprint written to "
        f"[bold]{cfg.fingerprint_path}[/bold]"
    )
    _render_fingerprint(console, fp, cfg.fingerprint_path)
    return 0


def run_style_update(cfg: Config, console: Console) -> int:
    """Recompute the fingerprint from samples + accepted Scribe revisions.

    Acts as the continuous-learning entry point (M4 S4.1.3): every
    interactive review session that the user accepted now feeds the
    fingerprint update. The previous ``fingerprint.json`` is archived
    under ``fingerprint_vN.json`` before the new one lands.
    """
    db = Database(cfg.db_path)
    try:
        sample_repo = StyleSampleRepository(db)
        rev_repo = DraftRevisionRepository(db)
        base_samples = sample_repo.list_all()
        revisions = rev_repo.list_all()
    finally:
        db.close()

    extra_samples = []
    for rev in revisions:
        extra_samples.extend(
            extract_samples_from_text(
                rev.revised_text,
                paper_id=f"revision:{rev.id}",
                section_title=rev.section,
            )
        )

    combined = base_samples + extra_samples
    if not combined:
        console.print(
            "[yellow]No samples to learn from.[/yellow] "
            "Train first with [bold]research style train[/bold]."
        )
        return 1

    fp = StyleAnalyzer().analyze(combined)
    fp.save_to(cfg.fingerprint_path, preserve_history=True)
    console.print(
        f"[green]✓[/green] Updated fingerprint v{fp.version} → "
        f"[bold]{cfg.fingerprint_path}[/bold]\n"
        f"[dim]Base samples: {len(base_samples)}, "
        f"revision-derived: {len(extra_samples)} "
        f"(from {len(revisions)} accepted revisions)[/dim]"
    )
    _render_fingerprint(console, fp, cfg.fingerprint_path)
    return 0


def run_style_history(cfg: Config, console: Console) -> int:
    """List archived fingerprint versions next to the current one."""
    style_dir = cfg.style_dir
    if not style_dir.exists():
        console.print(
            "[yellow]No fingerprints exist yet.[/yellow] "
            "Run [bold]research style fingerprint[/bold] first."
        )
        return 0
    archives = sorted(style_dir.glob("fingerprint_v*.json"))
    current = cfg.fingerprint_path
    table = Table(title="Fingerprint history", show_header=True)
    table.add_column("Version", justify="right")
    table.add_column("Path")
    table.add_column("Created")
    table.add_column("Samples", justify="right")
    rows: list[tuple[int, Path]] = []
    if current.exists():
        try:
            cur_fp = Fingerprint.load_from(current)
            rows.append((cur_fp.version, current))
        except Exception:
            pass
    for path in archives:
        try:
            fp = Fingerprint.load_from(path)
            rows.append((fp.version, path))
        except Exception:
            continue
    if not rows:
        console.print("[yellow]No readable fingerprint files yet.[/yellow]")
        return 0
    rows.sort(key=lambda kv: kv[0], reverse=True)
    for version, path in rows:
        try:
            fp = Fingerprint.load_from(path)
        except Exception:
            continue
        is_current = path == current
        marker = " [bold green](current)[/bold green]" if is_current else ""
        table.add_row(
            f"v{version}{marker}",
            str(path.name),
            fp.created_at or "—",
            f"{fp.sample_count} ({fp.paper_count}p)",
        )
    console.print(table)
    return 0


def _render_fingerprint(  # type: ignore[no-untyped-def]
    console: Console, fp: Fingerprint, path
) -> None:
    table = Table(title=f"Style fingerprint  ({path.name})", show_header=True)
    table.add_column("Layer", style="cyan")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("meta", "papers / samples", f"{fp.paper_count} / {fp.sample_count}")
    table.add_row("meta", "created_at", fp.created_at or "—")
    table.add_row("macro", "abstract opener", fp.macro.abstract_opener or "—")
    table.add_row("macro", "abstract avg sents", f"{fp.macro.abstract_avg_sentences:.1f}")
    table.add_row("macro", "intro opener", fp.macro.intro_opener or "—")
    table.add_row("macro", "intro avg paragraphs", f"{fp.macro.intro_avg_paragraphs:.1f}")
    table.add_row("macro", "related-work strategy", fp.macro.related_work_strategy or "—")
    table.add_row("macro", "sections/paper avg", f"{fp.macro.section_count_avg:.1f}")
    table.add_row("micro", "avg sentence length (words)", f"{fp.micro.avg_sentence_length:.1f}")
    table.add_row("micro", "median sentence length", f"{fp.micro.median_sentence_length:.1f}")
    table.add_row(
        "micro",
        "p10 / p90 sentence",
        f"{fp.micro.p10_sentence_length:.1f} / {fp.micro.p90_sentence_length:.1f}",
    )
    table.add_row("micro", "avg paragraph (sents)", f"{fp.micro.avg_paragraph_length:.1f}")
    table.add_row("micro", "hedging / 100 sents", f"{fp.micro.hedging_per_100:.1f}")
    table.add_row("micro", "confidence / 100 sents", f"{fp.micro.confidence_per_100:.1f}")
    table.add_row("micro", "passive / 100 sents", f"{fp.micro.passive_per_100:.1f}")
    table.add_row("micro", "type-token ratio", f"{fp.micro.type_token_ratio:.3f}")
    top_transitions = sorted(
        fp.micro.transition_freq.items(), key=lambda kv: kv[1], reverse=True
    )[:5]
    if top_transitions:
        table.add_row(
            "micro",
            "top transitions",
            ", ".join(f"{w} ({v:.1f})" for w, v in top_transitions),
        )
    table.add_row("markers", "citation format", fp.markers.citation_format or "—")
    table.add_row("markers", "figure refs", fp.markers.figure_ref_format or "—")
    table.add_row("markers", "table refs", fp.markers.table_ref_format or "—")
    table.add_row("markers", "em-dash / 100 sents", f"{fp.markers.em_dash_per_100:.1f}")
    if fp.markers.top_section_titles:
        table.add_row(
            "markers",
            "top section titles",
            ", ".join(fp.markers.top_section_titles[:5]),
        )
    console.print(table)
