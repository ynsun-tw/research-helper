"""Unit tests for :mod:`research_agent.style.samples`."""

from __future__ import annotations

from research_agent.storage.database import Database
from research_agent.style.samples import StyleSample, StyleSampleRepository


def _sample(paper_id: str, paragraph: str = "Hello world.", section: str = "Intro") -> StyleSample:
    return StyleSample(
        id="",
        paper_id=paper_id,
        section_title=section,
        paragraph=paragraph,
        char_count=len(paragraph),
        word_count=len(paragraph.split()),
        sentence_count=1,
    )


def test_add_and_list(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = StyleSampleRepository(db)
    s = repo.add(
        paper_id="arxiv:0000.0001",
        section_title="Intro",
        paragraph="A short prose paragraph for the test.",
        char_count=40,
        word_count=7,
        sentence_count=1,
    )
    assert s.id
    rows = repo.list_all()
    assert len(rows) == 1
    assert rows[0].paper_id == "arxiv:0000.0001"
    assert rows[0].section_title == "Intro"


def test_bulk_add(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = StyleSampleRepository(db)
    n = repo.bulk_add([_sample("p1"), _sample("p1"), _sample("p2")])
    assert n == 3
    assert repo.count() == 3
    counts = repo.count_by_paper()
    assert counts == {"p1": 2, "p2": 1}


def test_for_paper_and_iter(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = StyleSampleRepository(db)
    repo.bulk_add([_sample("a"), _sample("b"), _sample("a")])
    a_rows = repo.for_paper("a")
    assert len(a_rows) == 2
    streamed = list(repo.iter_all())
    assert len(streamed) == 3


def test_delete_for_paper(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = StyleSampleRepository(db)
    repo.bulk_add([_sample("a"), _sample("a"), _sample("b")])
    removed = repo.delete_for_paper("a")
    assert removed == 2
    assert repo.count() == 1
    assert repo.count_by_paper() == {"b": 1}


def test_bulk_add_empty_is_noop(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = StyleSampleRepository(db)
    assert repo.bulk_add([]) == 0
    assert repo.count() == 0
