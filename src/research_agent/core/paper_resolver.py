"""Resolve a user paper query to a loaded :class:`Paper`."""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

from rich.console import Console
from rich.table import Table

from research_agent.core.loader import PaperLoadError, load_paper
from research_agent.core.paper import Paper
from research_agent.search.arxiv import normalize_arxiv_id
from research_agent.search.arxiv_search import ArxivSearcher, ArxivSearchError, ArxivSearchHit
from research_agent.search.semantic_scholar import SemanticScholarSearcher


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
    fallback_searcher: SemanticScholarSearcher | None = None,
    max_results: int = 5,
    use_fallback: bool = True,
    merge_sources: bool = False,
) -> list[ArxivSearchHit]:
    """Search by title/keywords with arXiv primary, Semantic Scholar fallback.

    Default (``merge_sources=False``) - sequential, mutually exclusive:
    - arXiv returns >= 1 hit  -> use those.
    - arXiv returns 0 hits     -> try Semantic Scholar (fallback).
    - arXiv raises             -> try Semantic Scholar (fallback).
    - Both raise / empty       -> raise PaperLoadError mentioning both.

    ``merge_sources=True`` (M3 T3.1.2.3) - query *both* sources in
    parallel-of-intent, merge, and dedupe by arXiv id (after version
    stripping) with a normalised title-similarity fallback. arXiv hits
    win ties so users see arXiv-source attribution when a paper is in
    both indexes. Failure of one source no longer aborts the search;
    only an all-empty / all-error result raises.
    """
    stripped = query.strip()
    if not stripped:
        raise PaperLoadError(
            "Paper query is empty. Use --paper with an arXiv id, title, or PDF path."
        )
    primary = searcher or ArxivSearcher()
    fallback = fallback_searcher or SemanticScholarSearcher()

    if merge_sources:
        return _merged_search(
            stripped, primary, fallback, max_results=max_results
        )

    primary_err: ArxivSearchError | None = None
    try:
        hits = primary.search(stripped, max_results=max_results)
        if hits:
            return hits
    except ArxivSearchError as exc:
        primary_err = exc

    if not use_fallback:
        if primary_err is not None:
            raise PaperLoadError(str(primary_err)) from primary_err
        raise PaperLoadError(f"No arXiv papers found for: {stripped!r}")

    try:
        fallback_hits = fallback.search(stripped, max_results=max_results)
    except ArxivSearchError as fb_exc:
        msg = f"Semantic Scholar fallback also failed: {fb_exc}"
        if primary_err is not None:
            msg = f"arXiv failed ({primary_err}). {msg}"
        raise PaperLoadError(msg) from fb_exc

    if fallback_hits:
        return fallback_hits
    if primary_err is not None:
        raise PaperLoadError(
            f"arXiv failed ({primary_err}); Semantic Scholar returned no hits."
        ) from primary_err
    raise PaperLoadError(f"No papers found for: {stripped!r} (tried arXiv + S2)")


def _merged_search(
    query: str,
    primary: ArxivSearcher,
    fallback: SemanticScholarSearcher,
    *,
    max_results: int,
) -> list[ArxivSearchHit]:
    """Query both sources, merge, dedupe. Tolerant of single-source failure."""
    arxiv_hits: list[ArxivSearchHit] = []
    s2_hits: list[ArxivSearchHit] = []
    errs: list[str] = []
    try:
        arxiv_hits = primary.search(query, max_results=max_results)
    except ArxivSearchError as exc:
        errs.append(f"arXiv: {exc}")
    try:
        s2_hits = fallback.search(query, max_results=max_results)
    except ArxivSearchError as exc:
        errs.append(f"Semantic Scholar: {exc}")

    merged = dedupe_hits(arxiv_hits, s2_hits)
    if merged:
        return merged[:max_results] if max_results > 0 else merged
    if errs:
        raise PaperLoadError(
            f"Merged search failed for {query!r}: " + "; ".join(errs)
        )
    raise PaperLoadError(f"No papers found for: {query!r} (tried arXiv + S2)")


# ----------------------------------------------------- dedupe helpers (T3.1.2.3)


# Match "v1", "v23" version suffix on arXiv ids like 1706.03762v5.
_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^\w\s]")


def _strip_version(arxiv_id: str) -> str:
    """Normalise an arXiv id for equality comparison.

    "1706.03762" and "1706.03762v5" should hash the same so a hit from
    arXiv (often versionless) and S2 (often versioned) doesn't appear
    twice in merged results.
    """
    return _VERSION_SUFFIX_RE.sub("", arxiv_id.strip()).lower()


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - for fuzzy dedup."""
    cleaned = _NON_WORD_RE.sub(" ", title.lower())
    return " ".join(cleaned.split())


def _titles_similar(a: str, b: str, *, threshold: float = 0.9) -> bool:
    """Defensive title check for cases where two sources have different
    arXiv ids for the same paper (e.g., a withdrawn-then-resubmitted
    paper got a new id). Uses normalised strings + SequenceMatcher
    ratio - O(len*len) but inputs are short (paper titles) and dedup
    runs at most a few dozen times per search.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def dedupe_hits(*hit_lists: list[ArxivSearchHit]) -> list[ArxivSearchHit]:
    """Merge multiple hit lists, preferring earlier lists on collisions.

    Primary key: arXiv id with version suffix stripped. Fallback key:
    normalised-title similarity above 0.9. Returns merged list with
    duplicates removed; original order within each input list is
    preserved.

    >>> from research_agent.search.arxiv_search import ArxivSearchHit
    >>> a = ArxivSearchHit("1706.03762", "Attention Is All You Need", "")
    >>> b = ArxivSearchHit(
    ...     "1706.03762v5", "Attention Is All You Need", "",
    ...     source="semantic_scholar",
    ... )
    >>> deduped = dedupe_hits([a], [b])
    >>> [h.source for h in deduped]
    ['arxiv']
    """
    seen_ids: set[str] = set()
    seen_titles: list[str] = []
    result: list[ArxivSearchHit] = []
    for source_list in hit_lists:
        for hit in source_list:
            key = _strip_version(hit.arxiv_id) if hit.arxiv_id else ""
            if key and key in seen_ids:
                continue
            ntitle = _normalize_title(hit.title)
            if ntitle and any(_titles_similar(ntitle, t) for t in seen_titles):
                continue
            if key:
                seen_ids.add(key)
            if ntitle:
                seen_titles.append(ntitle)
            result.append(hit)
    return result


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
