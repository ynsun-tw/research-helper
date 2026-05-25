"""Unit tests for ``core.loader.load_paper``."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.core.loader import PaperLoadError, load_paper
from research_agent.search.arxiv import ArxivFetcher


def test_load_local_pdf(sample_pdf: Path, tmp_path: Path) -> None:
    paper = load_paper(str(sample_pdf), cache_dir=tmp_path / "cache")
    assert paper.id.startswith("local:")
    assert "Attention" in paper.title


def test_load_arxiv_uses_fetcher(
    sample_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"

    def fake_download(self: ArxivFetcher, url: str, dest: Path) -> None:
        dest.write_bytes(sample_pdf.read_bytes())

    monkeypatch.setattr(ArxivFetcher, "_download", fake_download)
    paper = load_paper("arxiv:2301.12345", cache_dir=cache)
    assert paper.id == "arxiv:2301.12345"
    assert (cache / "2301.12345.pdf").exists()


def test_load_missing_source(tmp_path: Path) -> None:
    with pytest.raises(PaperLoadError):
        load_paper("./does-not-exist.pdf", cache_dir=tmp_path)


def test_load_rejects_non_pdf(tmp_path: Path) -> None:
    other = tmp_path / "note.txt"
    other.write_text("hi")
    with pytest.raises(PaperLoadError):
        load_paper(str(other), cache_dir=tmp_path)
