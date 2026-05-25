"""Integration tests for ``Database`` + ``PaperRepository``."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.core.paper import Paper, Section
from research_agent.storage.database import Database, PaperRepository


@pytest.fixture
def repo(tmp_path: Path) -> PaperRepository:
    db = Database(tmp_path / "memory.db")
    return PaperRepository(db)


def _make_paper(**overrides: object) -> Paper:
    base = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="We propose the Transformer.",
        sections=[Section(title="1. Introduction", content="Recurrent models...")],
        full_text="full text here",
        year=2017,
        venue="NeurIPS",
        pdf_path=Path("/tmp/x.pdf"),
        tags=["transformer", "nlp"],
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_schema_creates_tables(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    names = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"papers", "ideas", "discussions"}.issubset(names)


def test_save_and_get_roundtrip(repo: PaperRepository) -> None:
    paper = _make_paper()
    repo.save(paper)
    loaded = repo.get(paper.id)
    assert loaded is not None
    assert loaded.title == paper.title
    assert loaded.authors == paper.authors
    assert loaded.tags == paper.tags
    assert loaded.sections[0].title == "1. Introduction"
    assert loaded.full_text == "full text here"
    assert loaded.year == 2017


def test_save_upserts_on_conflict(repo: PaperRepository) -> None:
    repo.save(_make_paper())
    repo.save(_make_paper(title="Attention Is All You Need v2", tags=["transformer"]))
    loaded = repo.get("arxiv:1706.03762")
    assert loaded is not None
    assert loaded.title.endswith("v2")
    assert loaded.tags == ["transformer"]


def test_get_missing_returns_none(repo: PaperRepository) -> None:
    assert repo.get("arxiv:does-not-exist") is None


def test_find_by_title(repo: PaperRepository) -> None:
    repo.save(_make_paper())
    repo.save(_make_paper(id="arxiv:2010.11929", title="An Image Is Worth 16x16 Words"))
    results = repo.find_by_title("Attention")
    assert len(results) == 1
    assert results[0].id == "arxiv:1706.03762"


def test_find_by_tag(repo: PaperRepository) -> None:
    repo.save(_make_paper())
    repo.save(_make_paper(id="arxiv:2010.11929", title="ViT", tags=["vision"]))
    nlp = repo.find_by_tag("nlp")
    assert [p.id for p in nlp] == ["arxiv:1706.03762"]
    vision = repo.find_by_tag("vision")
    assert [p.id for p in vision] == ["arxiv:2010.11929"]


def test_list_all(repo: PaperRepository) -> None:
    repo.save(_make_paper())
    repo.save(_make_paper(id="arxiv:2010.11929", title="ViT"))
    assert {p.id for p in repo.list_all()} == {"arxiv:1706.03762", "arxiv:2010.11929"}


def test_delete(repo: PaperRepository) -> None:
    repo.save(_make_paper())
    assert repo.delete("arxiv:1706.03762") is True
    assert repo.get("arxiv:1706.03762") is None
    assert repo.delete("arxiv:1706.03762") is False
