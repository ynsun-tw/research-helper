"""Live integration test for the S2 citation graph (M3 T3.1.2.4).

Hits the real Semantic Scholar API for a stable, well-known paper and
verifies the data we parse is structurally complete enough for /cites
and /refs to be useful. Skipped by default - opt in with::

    RUN_NETWORK_TESTS=1 pytest tests/integration/test_citation_graph_live.py

We use "Attention Is All You Need" (arXiv:1706.03762) as the anchor; it
has thousands of citations and references, so the assertions are loose
("at least N hits", "all entries structurally valid", "ids look like
arXiv ids", "no duplicates within a response").

Network-tolerance: S2's unauthenticated traffic is rate-limited
(~100 req / 5 min per IP). If we get 429 / timeout during a test run,
the test is **skipped** rather than failed; the assertion focus is on
"when S2 answers, is the data we surface correct?" not "is S2 always
reachable from this CI runner?".
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TypeVar

import pytest

from research_agent.search.arxiv_search import ArxivSearchError, ArxivSearchHit
from research_agent.search.semantic_scholar import SemanticScholarSearcher

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.environ.get("RUN_NETWORK_TESTS"),
        reason="network test; set RUN_NETWORK_TESTS=1 to enable",
    ),
]

ATTENTION_ARXIV_ID = "1706.03762"

# Modern arXiv ids look like YYYY.NNNNN (with optional v<N> suffix).
# Old format (cs.CL/0001001) is also valid; we accept either.
ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,6}(v\d+)?|[a-z\-]+(\.[A-Z]{2})?/\d{7})$")

T = TypeVar("T")


def _safe_call(func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run ``func``; skip the test on transient S2 failures (429 / timeout)."""
    try:
        return func(*args, **kwargs)
    except ArxivSearchError as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ("rate-limit", "429", "timed out")):
            pytest.skip(f"S2 transient failure: {exc}")
        raise


@pytest.fixture(scope="module")
def s2() -> SemanticScholarSearcher:
    # Honour S2 / arXiv TOS - 3s gap between requests (the default).
    return SemanticScholarSearcher()


def _assert_hit_is_structurally_valid(hit: ArxivSearchHit, *, tag: str) -> None:
    """Every returned hit must be downstream-usable (loader + queue + UI)."""
    assert isinstance(hit.arxiv_id, str), f"{tag}: arxiv_id not str"
    assert hit.arxiv_id.strip(), f"{tag}: arxiv_id empty"
    assert ARXIV_ID_RE.match(hit.arxiv_id), (
        f"{tag}: arxiv_id {hit.arxiv_id!r} doesn't look like a valid id"
    )
    assert hit.title.strip(), f"{tag}: title empty for {hit.arxiv_id}"
    assert hit.source == "semantic_scholar", (
        f"{tag}: wrong source ({hit.source}) for {hit.arxiv_id}"
    )


def test_s2_search_recovers_attention_paper(s2: SemanticScholarSearcher) -> None:
    hits = _safe_call(s2.search, "attention is all you need transformer", max_results=5)
    assert hits, "S2 returned no hits for the canonical query"
    assert any(h.arxiv_id == ATTENTION_ARXIV_ID for h in hits), (
        f"Expected arXiv:{ATTENTION_ARXIV_ID} in top-5 search results; "
        f"got {[h.arxiv_id for h in hits]}"
    )
    for h in hits:
        _assert_hit_is_structurally_valid(h, tag="search")


def test_s2_get_citations_returns_structurally_valid_arxiv_hits(
    s2: SemanticScholarSearcher,
) -> None:
    """Forward references: when S2 has arXiv-mapped citations, they must be
    structurally valid.

    Note: S2's first page of citations for a popular paper is often a
    mix of modern journal/conference papers without arXiv mappings.
    Our parser filters those out (downstream /read needs an arXiv id),
    so the *count* of returned hits depends on the underlying mix and
    we don't assert a minimum. The data we *do* return must be sound.
    """
    hits = _safe_call(s2.get_citations, ATTENTION_ARXIV_ID, max_results=25)
    for h in hits:
        _assert_hit_is_structurally_valid(h, tag="citations")
    ids = [h.arxiv_id for h in hits]
    assert len(ids) == len(set(ids)), f"Duplicate citations: {ids}"


def test_s2_get_references_returns_structurally_valid_arxiv_hits(
    s2: SemanticScholarSearcher,
) -> None:
    """Backward references: AIAYN's bibliography exposed via S2.

    We don't assert any *specific* paper appears - S2's reference list
    is paginated and the order isn't documented as relevance-sorted, so
    a "must include 1409.0473" check would be brittle. Instead we verify
    the data we'd surface to /refs / get_references is usable.
    """
    hits = _safe_call(s2.get_references, ATTENTION_ARXIV_ID, max_results=25)
    assert len(hits) >= 5, f"Expected >= 5 references, got {len(hits)}"
    for h in hits:
        _assert_hit_is_structurally_valid(h, tag="references")
    ids = [h.arxiv_id for h in hits]
    assert len(ids) == len(set(ids)), f"Duplicate references: {ids}"


def test_get_citations_paginates_via_max_results(
    s2: SemanticScholarSearcher,
) -> None:
    """max_results should be honoured by the server (no over-fetching)."""
    hits = _safe_call(s2.get_citations, ATTENTION_ARXIV_ID, max_results=3)
    # AIAYN has thousands of citations, so the cap should always bite.
    assert len(hits) <= 3
    for h in hits:
        _assert_hit_is_structurally_valid(h, tag="cap")


def test_round_trip_anchor_paper_appears_in_search_then_reference_graph(
    s2: SemanticScholarSearcher,
) -> None:
    """One round-trip check: an id returned by search() must be usable by
    get_references(). Guards against subtle id-format mismatches
    between endpoints (e.g., if search ever returned 1706.03762v5
    while /references expected 1706.03762).

    We use get_references rather than get_citations because S2's
    citation feed for popular papers is dominated by journal entries
    without arXiv mappings - which our filter drops to 0 - whereas
    AIAYN's bibliography is heavy with arXiv-published prior work.
    """
    search_hits = _safe_call(
        s2.search, "attention is all you need transformer", max_results=3
    )
    target = next((h for h in search_hits if h.arxiv_id == ATTENTION_ARXIV_ID), None)
    assert target is not None, "search did not surface the anchor paper"
    ref_hits = _safe_call(s2.get_references, target.arxiv_id, max_results=5)
    assert ref_hits, "get_references returned nothing for a known anchor paper"
    for h in ref_hits:
        _assert_hit_is_structurally_valid(h, tag="round-trip")
