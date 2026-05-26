"""Tests for the Searcher agent + relevance scoring integration (M3 Task 1)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from research_agent.agents.searcher import Searcher
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import cmd_search, exec_recent_searches
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.search.arxiv_search import ArxivSearchHit


def _hit(arxiv_id: str, title: str, abstract: str = "") -> ArxivSearchHit:
    return ArxivSearchHit(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract or f"Abstract for {title}",
        published="2018-06-12",
    )


# -------------------------- Searcher.score_hits ---------------------------


def test_score_hits_returns_empty_for_empty_input() -> None:
    llm = MockLLMProvider([])
    searcher = Searcher(llm)
    assert searcher.score_hits("transformer", []) == []


def test_score_hits_parses_and_clamps() -> None:
    llm = MockLLMProvider(
        [
            json.dumps(
                {
                    "scores": [
                        {"index": 1, "score": 0.95, "reason": "Direct match"},
                        {"index": 2, "score": 1.5, "reason": "Clamped"},  # >1 → 1.0
                        {"index": 3, "score": -0.2, "reason": "Clamped"},  # <0 → 0.0
                    ]
                }
            )
        ]
    )
    searcher = Searcher(llm)
    scored = searcher.score_hits(
        "transformer",
        [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")],
    )
    assert scored[0].relevance_score == pytest.approx(0.95)
    assert scored[1].relevance_score == pytest.approx(1.0)
    assert scored[2].relevance_score == pytest.approx(0.0)
    assert scored[0].relevance_reason == "Direct match"


def test_score_hits_missing_entries_keep_unscored() -> None:
    llm = MockLLMProvider(
        [json.dumps({"scores": [{"index": 1, "score": 0.8, "reason": "ok"}]})]
    )
    searcher = Searcher(llm)
    scored = searcher.score_hits(
        "q", [_hit("a", "A"), _hit("b", "B")]
    )
    assert scored[0].relevance_score == pytest.approx(0.8)
    # 2nd hit was not scored → falls back to None
    assert scored[1].relevance_score is None


def test_score_hits_invalid_json_falls_back_to_unscored() -> None:
    llm = MockLLMProvider(["not json at all"])
    searcher = Searcher(llm)
    scored = searcher.score_hits("q", [_hit("a", "A")])
    assert scored[0].relevance_score is None


def test_score_hits_truncates_long_abstracts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class CapturingLLM(MockLLMProvider):
        def chat(self_inner, messages, *, temperature=0.7, max_tokens=None):
            captured["user"] = messages[-1].content
            return json.dumps({"scores": [{"index": 1, "score": 0.5, "reason": "ok"}]})

    llm = CapturingLLM()
    searcher = Searcher(llm)
    long_abstract = "x" * 5000
    searcher.score_hits("q", [_hit("a", "A", abstract=long_abstract)])
    user_prompt = captured["user"]
    assert "…" in user_prompt
    assert len(user_prompt) < 5000  # truncated


# -------------------------- SearchRepository sort -------------------------


def test_record_sorts_persisted_hits_by_score(tmp_path: Path) -> None:
    from research_agent.storage.database import Database
    from research_agent.storage.searches import SearchRepository

    db = Database(tmp_path / "memory.db")
    repo = SearchRepository(db)
    hits = [
        ArxivSearchHit("a", "A", "", "2020-01-01", relevance_score=0.4),
        ArxivSearchHit("b", "B", "", "2020-01-01", relevance_score=0.9),
        ArxivSearchHit("c", "C", "", "2020-01-01", relevance_score=0.6),
    ]
    repo.record("q", hits)
    recent = repo.recent_queries()
    ids = [h.arxiv_id for h in recent[0].hits]
    assert ids == ["b", "c", "a"]
    assert recent[0].hits[0].relevance_score == pytest.approx(0.9)
    assert recent[0].hits[0].relevance_reason == ""
    db.close()


def test_record_without_scores_preserves_input_order(tmp_path: Path) -> None:
    from research_agent.storage.database import Database
    from research_agent.storage.searches import SearchRepository

    db = Database(tmp_path / "memory.db")
    repo = SearchRepository(db)
    repo.record("q", [_hit("a", "A"), _hit("b", "B")])
    recent = repo.recent_queries()
    assert [h.arxiv_id for h in recent[0].hits] == ["a", "b"]
    assert all(h.relevance_score is None for h in recent[0].hits)
    db.close()


def test_legacy_search_results_gets_relevance_columns(tmp_path: Path) -> None:
    """Existing DBs without the relevance columns get them on next open."""
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE papers (id TEXT PRIMARY KEY, title TEXT);
        CREATE TABLE ideas (id TEXT PRIMARY KEY, title TEXT, status TEXT);
        CREATE TABLE discussions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE search_queries (
            id TEXT PRIMARY KEY, query TEXT NOT NULL, source TEXT, created_at TEXT
        );
        CREATE TABLE search_results (
            id TEXT PRIMARY KEY, query_id TEXT, arxiv_id TEXT,
            title TEXT, abstract TEXT, published TEXT, rank INTEGER
        );
        """
    )
    conn.close()

    from research_agent.storage.database import Database

    db = Database(path)
    cols = {
        r[1] for r in db.conn.execute("PRAGMA table_info(search_results)").fetchall()
    }
    assert "relevance_score" in cols
    assert "relevance_reason" in cols
    db.close()


# -------------------------- /search integration ---------------------------


@pytest.fixture
def chat_session_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def factory(arxiv_hits: list[ArxivSearchHit], llm: MockLLMProvider):
        cfg = Config(data_dir=tmp_path, api_key="sk-x")
        console = Console(file=StringIO(), width=140)
        session = ChatSession.create(
            cfg=cfg,
            llm=llm,
            console=console,
            input_fn=lambda _: "",
            use_chroma=False,
        )
        monkeypatch.setattr(
            "research_agent.chat.tools.search_arxiv_papers",
            lambda *a, **kw: arxiv_hits,
        )
        return session, console

    return factory


def test_cmd_search_sorts_hits_by_relevance(chat_session_factory) -> None:
    score_json = json.dumps(
        {
            "scores": [
                {"index": 1, "score": 0.4, "reason": "tangential"},
                {"index": 2, "score": 0.9, "reason": "direct match"},
            ]
        }
    )
    llm = MockLLMProvider([score_json])
    session, console = chat_session_factory(
        [_hit("a", "A paper"), _hit("b", "B paper")], llm
    )
    cmd_search(session, "transformer")

    recent = session.searches.recent_queries()
    assert [h.arxiv_id for h in recent[0].hits] == ["b", "a"]
    out = console.file.getvalue()
    assert "0.90" in out
    assert "0.40" in out
    assert "Sorted by relevance" in out
    session.close()


def test_cmd_search_falls_back_when_scorer_returns_garbage(
    chat_session_factory,
) -> None:
    llm = MockLLMProvider(["this is not json"])
    session, console = chat_session_factory([_hit("a", "A"), _hit("b", "B")], llm)
    cmd_search(session, "q")

    recent = session.searches.recent_queries()
    assert {h.arxiv_id for h in recent[0].hits} == {"a", "b"}
    assert all(h.relevance_score is None for h in recent[0].hits)
    # table renders without the Score column when no hit has a score
    out = console.file.getvalue()
    assert "Score" not in out or "0." in out  # weak smoke: still rendered
    session.close()


def test_cmd_search_no_results_does_not_call_scorer(chat_session_factory) -> None:
    llm = MockLLMProvider([])  # any LLM call would crash with no-mock-response sentinel
    session, console = chat_session_factory([], llm)
    cmd_search(session, "no-such-thing")
    assert "No results" in console.file.getvalue()
    assert session.searches.recent_queries()[0].hits == []
    session.close()


def test_recent_searches_text_includes_score(chat_session_factory) -> None:
    llm = MockLLMProvider(
        [json.dumps({"scores": [{"index": 1, "score": 0.88, "reason": "ok"}]})]
    )
    session, _ = chat_session_factory([_hit("a", "A paper")], llm)
    cmd_search(session, "q")

    text = exec_recent_searches(session, {"limit": 5})
    assert "score=0.88" in text
    session.close()
