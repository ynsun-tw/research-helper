"""Unit tests for the arXiv fetcher (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.search.arxiv import ArxivFetcher, ArxivFetchError, normalize_arxiv_id


def test_normalize_arxiv_id_accepts_prefix() -> None:
    assert normalize_arxiv_id("arxiv:2301.12345") == "2301.12345"


def test_normalize_arxiv_id_accepts_bare() -> None:
    assert normalize_arxiv_id("1706.03762") == "1706.03762"


def test_normalize_arxiv_id_accepts_version() -> None:
    assert normalize_arxiv_id("arxiv:2301.12345v2") == "2301.12345v2"


def test_normalize_arxiv_id_rejects_other() -> None:
    assert normalize_arxiv_id("./local.pdf") is None
    assert normalize_arxiv_id("doi:10.1/xyz") is None


def test_fetch_uses_cache_when_present(tmp_path: Path) -> None:
    fetcher = ArxivFetcher(cache_dir=tmp_path)
    cached = fetcher.cache_path("2301.12345")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"%PDF-1.4 fake")

    result = fetcher.fetch("arxiv:2301.12345")
    assert result == cached
    assert result.read_bytes() == b"%PDF-1.4 fake"


def test_fetch_downloads_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = ArxivFetcher(cache_dir=tmp_path)

    def fake_download(self: ArxivFetcher, url: str, dest: Path) -> None:
        assert "2301.12345" in url
        dest.write_bytes(b"%PDF-1.4 downloaded")

    monkeypatch.setattr(ArxivFetcher, "_download", fake_download)
    path = fetcher.fetch("arxiv:2301.12345")
    assert path.exists()
    assert path.read_bytes() == b"%PDF-1.4 downloaded"


def test_fetch_rejects_non_arxiv(tmp_path: Path) -> None:
    fetcher = ArxivFetcher(cache_dir=tmp_path)
    with pytest.raises(ArxivFetchError):
        fetcher.fetch("./paper.pdf")
