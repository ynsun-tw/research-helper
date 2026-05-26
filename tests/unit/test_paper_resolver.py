"""Paper resolver search, selection, and arXiv → Semantic Scholar fallback."""

from __future__ import annotations

import pytest

from research_agent.core.loader import PaperLoadError
from research_agent.core.paper_resolver import (
    _normalize_title,
    _strip_version,
    _titles_similar,
    apply_search_mode,
    dedupe_hits,
    parse_search_mode,
    search_arxiv_papers,
    select_arxiv_hit,
)
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


# ------------------ cross-source dedup utilities (T3.1.2.3) -------------


def test_strip_version_normalises_arxiv_id() -> None:
    assert _strip_version("1706.03762") == "1706.03762"
    assert _strip_version("1706.03762v5") == "1706.03762"
    assert _strip_version("1706.03762V12") == "1706.03762"
    assert _strip_version(" 1706.03762v1 ") == "1706.03762"


def test_normalize_title_strips_punctuation_and_case() -> None:
    norm = _normalize_title("Attention Is All You Need!")
    assert norm == "attention is all you need"
    assert _normalize_title("  BERT:  Pre-training... ") == "bert pre training"


def test_titles_similar_handles_minor_variation() -> None:
    assert _titles_similar(
        _normalize_title("Attention Is All You Need"),
        _normalize_title("Attention is all you need."),
    )
    # Punctuation/case-only differences are normalised out -> exact match.
    assert _titles_similar(
        _normalize_title("BERT: Pre-training Deep Bidirectional Transformers"),
        _normalize_title("bert pre training deep bidirectional transformers"),
    )
    # Unrelated titles must NOT collide.
    assert not _titles_similar(
        _normalize_title("Attention Is All You Need"),
        _normalize_title("Generative Adversarial Networks"),
    )


def test_titles_similar_empty_returns_false() -> None:
    assert not _titles_similar("", "anything")
    assert not _titles_similar("anything", "")


def test_dedupe_collapses_by_arxiv_id_ignoring_version() -> None:
    a = ArxivSearchHit("1706.03762", "Attention Is All You Need", "")
    b = ArxivSearchHit(
        "1706.03762v5",
        "Attention Is All You Need",
        "",
        source="semantic_scholar",
    )
    result = dedupe_hits([a], [b])
    assert len(result) == 1
    # Earlier list wins -> arXiv source attribution preserved.
    assert result[0].source == "arxiv"


def test_dedupe_collapses_by_title_when_ids_differ() -> None:
    """Two indexes occasionally have different arXiv ids for the same paper
    (e.g., a withdraw+resubmit); a fuzzy title match still dedupes them."""
    a = ArxivSearchHit("1234.56789", "Attention Is All You Need", "")
    b = ArxivSearchHit(
        "9999.99999",
        "Attention is all you need!",  # punctuation diff only
        "",
        source="semantic_scholar",
    )
    result = dedupe_hits([a], [b])
    assert len(result) == 1
    assert result[0].arxiv_id == "1234.56789"


def test_dedupe_keeps_distinct_papers() -> None:
    a = ArxivSearchHit("1706.03762", "Attention Is All You Need", "")
    b = ArxivSearchHit("1810.04805", "BERT", "", source="semantic_scholar")
    c = ArxivSearchHit("2005.14165", "GPT-3", "", source="semantic_scholar")
    result = dedupe_hits([a], [b, c])
    assert [h.arxiv_id for h in result] == ["1706.03762", "1810.04805", "2005.14165"]


def test_dedupe_preserves_order_within_each_list() -> None:
    arxiv = [
        ArxivSearchHit("1706.03762", "Attention", ""),
        ArxivSearchHit("1810.04805", "BERT", ""),
    ]
    s2 = [
        ArxivSearchHit("2005.14165", "GPT-3", "", source="semantic_scholar"),
        ArxivSearchHit("1706.03762v5", "Attention", "", source="semantic_scholar"),
    ]
    result = dedupe_hits(arxiv, s2)
    # arXiv list first (in order), then S2's unique tail.
    assert [h.arxiv_id for h in result] == [
        "1706.03762",
        "1810.04805",
        "2005.14165",
    ]


def test_dedupe_handles_hit_without_arxiv_id() -> None:
    """Defensive: even though current parsers always populate arxiv_id,
    dedupe shouldn't crash on missing values."""
    a = ArxivSearchHit("", "Mystery paper", "")
    b = ArxivSearchHit("", "Mystery paper", "", source="semantic_scholar")
    c = ArxivSearchHit("", "Different paper", "")
    result = dedupe_hits([a], [b], [c])
    # a and b dedupe via title; c is distinct.
    assert len(result) == 2


# ----------------------- merged-source search ---------------------------


def test_merge_sources_combines_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    arxiv = _StubSearcher(
        [
            ArxivSearchHit("1706.03762", "Attention Is All You Need", ""),
            ArxivSearchHit("1810.04805", "BERT", ""),
        ]
    )
    s2 = _StubSearcher(
        [
            ArxivSearchHit(
                "1706.03762v5",
                "Attention Is All You Need",
                "",
                source="semantic_scholar",
            ),
            ArxivSearchHit(
                "2005.14165", "GPT-3", "", source="semantic_scholar"
            ),
        ]
    )
    hits = search_arxiv_papers(
        "transformers", searcher=arxiv, fallback_searcher=s2, merge_sources=True
    )
    ids = [h.arxiv_id for h in hits]
    # AIAYN deduped (arXiv version wins), BERT + GPT-3 carried through.
    assert ids == ["1706.03762", "1810.04805", "2005.14165"]
    sources = {h.source for h in hits}
    assert sources == {"arxiv", "semantic_scholar"}


def test_merge_sources_tolerates_arxiv_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    arxiv = _StubSearcher(raises=ArxivSearchError("HTTP 429"))
    s2 = _StubSearcher(
        [
            ArxivSearchHit(
                "1706.03762", "AIAYN", "", source="semantic_scholar"
            ),
        ]
    )
    hits = search_arxiv_papers(
        "x", searcher=arxiv, fallback_searcher=s2, merge_sources=True
    )
    assert hits[0].source == "semantic_scholar"


def test_merge_sources_tolerates_s2_failure() -> None:
    arxiv = _StubSearcher([ArxivSearchHit("1706.03762", "AIAYN", "")])
    s2 = _StubSearcher(raises=ArxivSearchError("500 down"))
    hits = search_arxiv_papers(
        "x", searcher=arxiv, fallback_searcher=s2, merge_sources=True
    )
    assert [h.arxiv_id for h in hits] == ["1706.03762"]


def test_merge_sources_raises_when_both_fail() -> None:
    arxiv = _StubSearcher(raises=ArxivSearchError("HTTP 429"))
    s2 = _StubSearcher(raises=ArxivSearchError("500 down"))
    with pytest.raises(PaperLoadError) as exc_info:
        search_arxiv_papers(
            "x", searcher=arxiv, fallback_searcher=s2, merge_sources=True
        )
    msg = str(exc_info.value)
    assert "arXiv" in msg
    assert "Semantic Scholar" in msg


# ----------------------- search modes (T3.1.3.3) ---------------------


def test_parse_search_mode_none_or_empty_returns_noop() -> None:
    assert parse_search_mode(None).kind is None
    assert parse_search_mode("").kind is None
    assert parse_search_mode("   ").kind is None


def test_parse_search_mode_recognises_theoretical_and_applied() -> None:
    assert parse_search_mode("theoretical").kind == "theoretical"
    assert parse_search_mode("THEORETICAL").kind == "theoretical"
    assert parse_search_mode("  applied  ").kind == "applied"


def test_parse_search_mode_recognises_group_with_author() -> None:
    parsed = parse_search_mode("group:Andrej Karpathy")
    assert parsed.kind == "group"
    assert parsed.author == "Andrej Karpathy"


def test_parse_search_mode_group_without_author_is_warning() -> None:
    parsed = parse_search_mode("group:  ")
    assert parsed.kind is None
    assert "requires an author" in parsed.warning


def test_parse_search_mode_unknown_emits_warning() -> None:
    parsed = parse_search_mode("recreational")
    assert parsed.kind is None
    assert "Unknown" in parsed.warning
    assert "theoretical" in parsed.warning  # lists valid modes


def test_apply_search_mode_prepends_bias_for_theoretical() -> None:
    result = apply_search_mode("attention transformer", "theoretical")
    assert "theoretical" in result.lower()
    assert "attention transformer" in result
    # Bias prepended, not replacing user keywords.
    assert result.endswith("attention transformer")


def test_apply_search_mode_prepends_bias_for_applied() -> None:
    result = apply_search_mode("attention transformer", "applied")
    assert "empirical" in result.lower() or "benchmark" in result.lower()
    assert "attention transformer" in result


def test_apply_search_mode_prepends_author_for_group() -> None:
    result = apply_search_mode("attention", "group:Andrej Karpathy")
    assert "Andrej Karpathy" in result
    assert "attention" in result


def test_apply_search_mode_unknown_is_noop() -> None:
    assert apply_search_mode("attention", "recreational") == "attention"


def test_apply_search_mode_none_is_noop() -> None:
    assert apply_search_mode("attention", None) == "attention"


def test_search_papers_passes_mode_to_searcher() -> None:
    """The mode bias must reach the underlying searcher as part of the query."""
    arxiv = _StubSearcher([ArxivSearchHit("1706.03762", "x", "")])
    search_arxiv_papers(
        "attention transformer", searcher=arxiv, mode="theoretical"
    )
    assert arxiv.calls, "searcher was not called"
    forwarded_query, _ = arxiv.calls[0]
    assert "theoretical" in forwarded_query.lower()
    assert "attention transformer" in forwarded_query


def test_search_papers_mode_group_passes_author() -> None:
    arxiv = _StubSearcher([ArxivSearchHit("1.1", "x", "")])
    search_arxiv_papers(
        "rnn", searcher=arxiv, mode="group:Yoshua Bengio"
    )
    forwarded_query, _ = arxiv.calls[0]
    assert "Yoshua Bengio" in forwarded_query


def test_merge_sources_caps_to_max_results() -> None:
    # Distinct titles to prevent the fuzzy title check from collapsing them.
    distinct_a = [
        "Attention Is All You Need",
        "BERT pretraining",
        "GPT-3 few-shot learners",
        "AlphaGo Zero",
        "ResNet deep residual learning",
    ]
    distinct_b = [
        "Word2Vec distributed representations",
        "ELMo contextualised embeddings",
        "T5 text-to-text transfer",
        "Adam optimisation",
        "Dropout regularisation",
    ]
    arxiv = _StubSearcher(
        [
            ArxivSearchHit(f"a.{i:05d}", title, "")
            for i, title in enumerate(distinct_a)
        ]
    )
    s2 = _StubSearcher(
        [
            ArxivSearchHit(
                f"s.{i:05d}", title, "", source="semantic_scholar"
            )
            for i, title in enumerate(distinct_b)
        ]
    )
    hits = search_arxiv_papers(
        "x",
        searcher=arxiv,
        fallback_searcher=s2,
        merge_sources=True,
        max_results=3,
    )
    assert len(hits) == 3
