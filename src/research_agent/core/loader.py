"""High-level paper loader: resolves `arxiv:<id>` or local PDF path to a Paper."""

from __future__ import annotations

from pathlib import Path

from research_agent.core.paper import Paper
from research_agent.parsers.pdf import parse_pdf
from research_agent.search.arxiv import ArxivFetcher, normalize_arxiv_id


class PaperLoadError(Exception):
    """Raised when a paper source cannot be resolved."""


def load_paper(source: str, *, cache_dir: Path) -> Paper:
    """Resolve `source` to a Paper.

    `source` may be:
      - ``arxiv:2301.12345`` or ``2301.12345`` → fetched into ``cache_dir`` and parsed
      - a path to a local PDF
    """
    arxiv_id = normalize_arxiv_id(source)
    if arxiv_id is not None:
        fetcher = ArxivFetcher(cache_dir=cache_dir)
        pdf_path = fetcher.fetch(arxiv_id)
        return parse_pdf(pdf_path, paper_id=f"arxiv:{arxiv_id}")

    path = Path(source).expanduser()
    if not path.exists():
        raise PaperLoadError(
            f"Source not recognized: {source!r}. Use 'arxiv:<id>' or a path to a local PDF."
        )
    if path.suffix.lower() != ".pdf":
        raise PaperLoadError(f"Unsupported file type: {path.suffix} (expected .pdf)")
    return parse_pdf(path)
