"""Resolve a user paper query to a loaded :class:`Paper`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from research_agent.core.loader import PaperLoadError, load_paper
from research_agent.core.paper import Paper
from research_agent.search.arxiv import normalize_arxiv_id
from research_agent.search.arxiv_search import ArxivSearcher, ArxivSearchError, ArxivSearchHit


def try_direct_paper_load(query: str, *, cache_dir: Path) -> Paper | None:
    """Return a paper when ``query`` is a local PDF or arXiv id; else ``None``."""
    stripped = query.strip()
    if not stripped:
        raise PaperLoadError(
            "Paper query is empty. Use --paper with an arXiv id, title, or PDF path."
        )
    path = Path(stripped).expanduser()
    if path.suffix.lower() == ".pdf" and path.exists():
        return load_paper(str(path), cache_dir=cache_dir)
    if normalize_arxiv_id(stripped) is not None:
        return load_paper(stripped, cache_dir=cache_dir)
    return None


def search_arxiv_papers(
    query: str,
    *,
    searcher: ArxivSearcher | None = None,
    max_results: int = 5,
) -> list[ArxivSearchHit]:
    """Search arXiv by title/keywords."""
    stripped = query.strip()
    if not stripped:
        raise PaperLoadError(
            "Paper query is empty. Use --paper with an arXiv id, title, or PDF path."
        )
    search = searcher or ArxivSearcher()
    try:
        hits = search.search(stripped, max_results=max_results)
    except ArxivSearchError as exc:
        raise PaperLoadError(str(exc)) from exc
    if not hits:
        raise PaperLoadError(f"No arXiv papers found for: {stripped!r}")
    return hits


def load_paper_from_hit(hit: ArxivSearchHit, *, cache_dir: Path) -> Paper:
    return load_paper(f"arxiv:{hit.arxiv_id}", cache_dir=cache_dir)


def resolve_paper_for_discuss(
    query: str,
    *,
    cache_dir: Path,
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
    searcher: ArxivSearcher | None = None,
) -> Paper:
    """Load a paper from arXiv id, local PDF path, or arXiv keyword search."""
    direct = try_direct_paper_load(query, cache_dir=cache_dir)
    if direct is not None:
        return direct
    hits = search_arxiv_papers(query, searcher=searcher)
    chosen = select_arxiv_hit(hits, console=console, input_fn=input_fn)
    return load_paper_from_hit(chosen, cache_dir=cache_dir)


def select_arxiv_hit(
    hits: list[ArxivSearchHit],
    *,
    console: Console | None,
    input_fn: Callable[[str], str] | None,
) -> ArxivSearchHit:
    if len(hits) == 1:
        return hits[0]

    if console is not None:
        console.print(
            "\n[bold]Multiple papers found on arXiv.[/bold] "
            "Choose the anchor paper for this discussion:"
        )
        table = Table(title="arXiv search results", show_header=True)
        table.add_column("#", style="dim", justify="right")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Year", justify="right")
        for i, hit in enumerate(hits, start=1):
            year = hit.published[:4] if hit.published else "—"
            table.add_row(str(i), hit.arxiv_id, hit.title[:80], year)
        console.print(table)

    if input_fn is None:
        if console is None:
            return hits[0]
        input_fn = console.input

    while True:
        raw = input_fn(f"Select paper [1-{len(hits)}] (default 1): ").strip()
        if not raw:
            return hits[0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(hits):
                return hits[idx - 1]
        if console is not None:
            console.print(f"[yellow]Enter a number from 1 to {len(hits)}.[/yellow]")
