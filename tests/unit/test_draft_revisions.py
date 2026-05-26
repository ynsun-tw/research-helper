"""Unit tests for :class:`DraftRevisionRepository` (M4 S4.3.2)."""

from __future__ import annotations

from research_agent.storage.database import Database
from research_agent.storage.draft_revisions import DraftRevisionRepository


def test_add_and_get(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = DraftRevisionRepository(db)
    rev = repo.add(
        section="introduction",
        original_text="First version.",
        revised_text="Better version.",
        selected_issues=["overclaim"],
        selected_suggestions=["cite Smith 2024"],
        rejected_issues=["too long"],
        rejected_suggestions=[],
        interactive=True,
    )
    assert rev.id
    loaded = repo.get(rev.id)
    assert loaded is not None
    assert loaded.section == "introduction"
    assert loaded.original_text == "First version."
    assert loaded.revised_text == "Better version."
    assert loaded.selected_issues == ["overclaim"]
    assert loaded.selected_suggestions == ["cite Smith 2024"]
    assert loaded.rejected_issues == ["too long"]
    assert loaded.rejected_suggestions == []
    assert loaded.interactive is True


def test_list_all_order(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = DraftRevisionRepository(db)
    for i in range(3):
        repo.add(
            section="abstract",
            original_text=f"v{i}",
            revised_text=f"v{i}+",
            interactive=False,
        )
    rows = repo.list_all()
    assert len(rows) == 3
    assert repo.count() == 3
    iterated = list(repo.iter_all())
    assert len(iterated) == 3


def test_get_missing(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = DraftRevisionRepository(db)
    assert repo.get("does-not-exist") is None


def test_empty_lists_default(config_dir) -> None:
    db = Database(config_dir / "memory.db")
    repo = DraftRevisionRepository(db)
    rev = repo.add(
        section="conclusion",
        original_text="a",
        revised_text="b",
    )
    loaded = repo.get(rev.id)
    assert loaded is not None
    assert loaded.selected_issues == []
    assert loaded.rejected_issues == []
    assert loaded.interactive is False
