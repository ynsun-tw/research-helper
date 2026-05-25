"""Unit tests for arXiv Atom search parsing."""

from __future__ import annotations

import pytest

from research_agent.search.arxiv_search import ArxivSearcher, _parse_atom_feed

SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>Sequence models based on complex recurrent networks.</summary>
    <published>2017-06-12T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.12345</id>
    <title>Another Paper</title>
    <summary>Abstract two.</summary>
    <published>2023-01-15T00:00:00Z</published>
  </entry>
</feed>
"""


def test_parse_atom_feed() -> None:
    hits = _parse_atom_feed(SAMPLE_FEED)
    assert len(hits) == 2
    assert hits[0].arxiv_id == "1706.03762v5"
    assert "Attention" in hits[0].title
    assert hits[0].published.startswith("2017")


def test_search_uses_http(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    hits = ArxivSearcher().search("attention transformer", max_results=5)
    assert hits[0].arxiv_id.startswith("1706")
