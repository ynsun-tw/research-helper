"""Tests for SearchRepository (M3.1 search history + dedup)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.core.paper import Paper
from research_agent.search.arxiv_search import ArxivSearchHit
from research_agent.storage.database import Database, PaperRepository
from research_agent.storage.searches import SearchRepository


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "memory.db")


def _hit(arxiv_id: str, title: str, year: str = "2017") -> ArxivSearchHit:
    return ArxivSearchHit(
        arxiv_id=arxiv_id,
        title=title,
        abstract=f"Abstract for {title}",
        published=f"{year}-06-12",
    )


def test_record_and_recent_queries_returns_hits_in_rank_order(tmp_path: Path) -> None:
    db = _db(tmp_path)
    repo = SearchRepository(db)

    hits = [_hit("1706.03762", "Attention Is All You Need"),
            _hit("1810.04805", "BERT")]
    qid = repo.record("transformer", hits, session_id="s1")
    assert qid

    recent = repo.recent_queries(limit=5)
    assert len(recent) == 1
    q = recent[0]
    assert q.query == "transformer"
    assert q.source == "arxiv"
    assert q.session_id == "s1"
    assert [h.arxiv_id for h in q.hits] == ["1706.03762", "1810.04805"]
    assert [h.rank for h in q.hits] == [1, 2]
    db.close()


def test_recent_queries_orders_newest_first(tmp_path: Path) -> None:
    import time

    db = _db(tmp_path)
    repo = SearchRepository(db)
    repo.record("first", [_hit("1.1", "First")])
    time.sleep(1.05)  # SQLite CURRENT_TIMESTAMP has second resolution
    repo.record("second", [_hit("2.2", "Second")])

    recent = repo.recent_queries()
    assert [q.query for q in recent] == ["second", "first"]
    db.close()


def test_already_read_detects_papers_with_or_without_arxiv_prefix(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    papers = PaperRepository(db)
    searches = SearchRepository(db)

    papers.save(Paper(id="arxiv:1706.03762", title="Attention", abstract="", full_text=""))
    papers.save(Paper(id="2010.11929", title="ViT", abstract="", full_text=""))

    seen = searches.already_read(["1706.03762", "2010.11929", "9999.99999"])
    assert seen == {"1706.03762", "2010.11929"}
    db.close()


def test_recent_queries_flags_read_hits(tmp_path: Path) -> None:
    db = _db(tmp_path)
    papers = PaperRepository(db)
    searches = SearchRepository(db)

    papers.save(Paper(id="arxiv:1706.03762", title="Attention", abstract="", full_text=""))
    searches.record(
        "transformer",
        [_hit("1706.03762", "Attention"), _hit("1810.04805", "BERT")],
    )

    recent = searches.recent_queries()
    assert len(recent) == 1
    hits = {h.arxiv_id: h.read for h in recent[0].hits}
    assert hits["1706.03762"] is True
    assert hits["1810.04805"] is False
    db.close()


def test_recent_queries_respects_limit(tmp_path: Path) -> None:
    db = _db(tmp_path)
    repo = SearchRepository(db)
    for i in range(5):
        repo.record(f"q{i}", [_hit(f"{i}.0", f"title {i}")])

    recent = repo.recent_queries(limit=3)
    assert len(recent) == 3
    db.close()


def test_recent_queries_empty(tmp_path: Path) -> None:
    db = _db(tmp_path)
    repo = SearchRepository(db)
    assert repo.recent_queries() == []
    db.close()


def test_record_with_no_hits(tmp_path: Path) -> None:
    db = _db(tmp_path)
    repo = SearchRepository(db)
    qid = repo.record("empty query", [])
    assert qid
    recent = repo.recent_queries()
    assert recent[0].hits == []
    db.close()


def test_database_creates_search_tables(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tables = {
        row["name"]
        for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "search_queries" in tables
    assert "search_results" in tables
    db.close()


def test_migration_adds_search_tables_to_legacy_db(tmp_path: Path) -> None:
    """Opening an existing DB without the search tables should auto-create them."""
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE papers (id TEXT PRIMARY KEY, title TEXT, abstract TEXT);
        CREATE TABLE ideas (id TEXT PRIMARY KEY, title TEXT, status TEXT);
        CREATE TABLE discussions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        """
    )
    conn.close()

    db = Database(path)
    repo = SearchRepository(db)
    qid = repo.record("transformer", [_hit("1706.03762", "Attention")])
    assert qid
    assert len(repo.recent_queries()) == 1
    db.close()


# -- /search + /history slash integration ----------------------------------


@pytest.fixture
def chat_session_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a ChatSession backed by a temp DB and a stubbed arXiv searcher."""
    from io import StringIO

    from rich.console import Console

    from research_agent.chat.session import ChatSession
    from research_agent.config import Config
    from research_agent.core.llm import MockLLMProvider

    def factory(hits: list[ArxivSearchHit]):
        cfg = Config(data_dir=tmp_path, api_key="sk-x")
        console = Console(file=StringIO(), width=120)
        session = ChatSession.create(
            cfg=cfg,
            llm=MockLLMProvider([]),
            console=console,
            input_fn=lambda _: "",
            use_chroma=False,
        )
        monkeypatch.setattr(
            "research_agent.chat.tools.search_arxiv_papers",
            lambda *a, **kw: hits,
        )
        return session, console

    return factory


def test_cmd_search_persists_query_and_flags_already_read(chat_session_factory) -> None:
    from research_agent.chat.tools import cmd_search

    hits = [_hit("1706.03762", "Attention"), _hit("1810.04805", "BERT")]
    session, console = chat_session_factory(hits)
    session.papers.save(
        Paper(id="arxiv:1706.03762", title="Attention", abstract="", full_text="")
    )

    cmd_search(session, "transformer")
    out = console.file.getvalue()
    # the green check renders as plain glyph ✓ in StringIO
    assert "1706.03762" in out
    assert "1810.04805" in out
    assert "✓" in out  # the already-read marker

    recent = session.searches.recent_queries()
    assert len(recent) == 1
    assert recent[0].query == "transformer"
    assert len(recent[0].hits) == 2
    session.close()


def test_cmd_history_shows_recent_searches(chat_session_factory) -> None:
    from research_agent.chat.tools import cmd_history, cmd_search

    session, console = chat_session_factory(
        [_hit("1706.03762", "Attention")]
    )
    cmd_search(session, "transformer")
    cmd_history(session, "")
    out = console.file.getvalue()
    assert "Recent searches" in out
    assert "transformer" in out
    session.close()


def test_cmd_history_empty_message(chat_session_factory) -> None:
    from research_agent.chat.tools import cmd_history

    session, console = chat_session_factory([])
    cmd_history(session, "")
    assert "No search history" in console.file.getvalue()
    session.close()


def test_cmd_history_bad_limit(chat_session_factory) -> None:
    from research_agent.chat.tools import cmd_history

    session, console = chat_session_factory([])
    cmd_history(session, "notanumber")
    assert "Usage" in console.file.getvalue()
    session.close()
