"""Paper resolver search, selection, and arXiv → Semantic Scholar fallback."""

from __future__ import annotations

import pytest

from research_agent.core.loader import PaperLoadError
from research_agent.core.paper_resolver import search_arxiv_papers, select_arxiv_hit
from research_agent.search.arxiv_search import ArxivSearchError, ArxivSearchHit


def test_select_arxiv_hit_accepts_choice() -> None:
    hits = [
        ArxivSearchHit("1111.1111", "Paper A", "abstract a"),
        ArxivSearchHit("2222.2222", "Paper B", "abstract b"),
    ]
    chosen = select_arxiv_hit(
        hits,
        console=None,
        input_fn=lambda _: "2",
    )
    assert chosen.arxiv_id == "2222.2222"


# ----------------------- arXiv → S2 fallback --------------------------


class _StubSearcher:
    """Minimal duck-type for ArxivSearcher / SemanticScholarSearcher."""

    def __init__(
        self,
        hits: list[ArxivSearchHit] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self._hits = hits or []
        self._raises = raises
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int = 5) -> list[ArxivSearchHit]:
        self.calls.append((query, max_results))
        if self._raises is not None:
            raise self._raises
        return list(self._hits)


def test_search_papers_uses_arxiv_when_it_succeeds() -> None:
    arxiv = _StubSearcher([ArxivSearchHit("1706.03762", "T", "abs")])
    fb = _StubSearcher([ArxivSearchHit("0000.0000", "S2", "abs", source="semantic_scholar")])
    hits = search_arxiv_papers(
        "transformer", searcher=arxiv, fallback_searcher=fb
    )
    assert [h.arxiv_id for h in hits] == ["1706.03762"]
    assert hits[0].source == "arxiv"
    assert fb.calls == []  # fallback never touched


def test_search_papers_falls_back_when_arxiv_raises() -> None:
    arxiv = _StubSearcher(raises=ArxivSearchError("HTTP 429"))
    fb = _StubSearcher(
        [ArxivSearchHit("1706.03762", "T (via S2)", "", source="semantic_scholar")]
    )
    hits = search_arxiv_papers("transformer", searcher=arxiv, fallback_searcher=fb)
    assert hits[0].source == "semantic_scholar"
    assert fb.calls == [("transformer", 5)]


def test_search_papers_falls_back_when_arxiv_returns_empty() -> None:
    arxiv = _StubSearcher([])
    fb = _StubSearcher(
        [ArxivSearchHit("9999.9999", "Match", "", source="semantic_scholar")]
    )
    hits = search_arxiv_papers("obscure topic", searcher=arxiv, fallback_searcher=fb)
    assert [h.arxiv_id for h in hits] == ["9999.9999"]


def test_search_papers_raises_when_both_sources_fail() -> None:
    arxiv = _StubSearcher(raises=ArxivSearchError("HTTP 429"))
    fb = _StubSearcher(raises=ArxivSearchError("upstream 500"))
    with pytest.raises(PaperLoadError) as exc_info:
        search_arxiv_papers("x", searcher=arxiv, fallback_searcher=fb)
    msg = str(exc_info.value)
    assert "arXiv failed" in msg
    assert "Semantic Scholar" in msg


def test_search_papers_raises_when_arxiv_raises_and_s2_empty() -> None:
    arxiv = _StubSearcher(raises=ArxivSearchError("HTTP 429"))
    fb = _StubSearcher([])
    with pytest.raises(PaperLoadError) as exc_info:
        search_arxiv_papers("x", searcher=arxiv, fallback_searcher=fb)
    assert "Semantic Scholar returned no hits" in str(exc_info.value)


def test_search_papers_opt_out_of_fallback() -> None:
    """use_fallback=False keeps the original arXiv-only behaviour."""
    arxiv = _StubSearcher(raises=ArxivSearchError("HTTP 429"))
    fb = _StubSearcher([ArxivSearchHit("x", "x", "")])
    with pytest.raises(PaperLoadError):
        search_arxiv_papers(
            "x", searcher=arxiv, fallback_searcher=fb, use_fallback=False
        )
    assert fb.calls == []


def test_search_papers_rejects_empty_query() -> None:
    with pytest.raises(PaperLoadError):
        search_arxiv_papers("   ")
