"""Paper resolver search and selection."""

from __future__ import annotations

from research_agent.core.paper_resolver import select_arxiv_hit
from research_agent.search.arxiv_search import ArxivSearchHit


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
