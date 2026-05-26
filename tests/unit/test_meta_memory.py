"""MetaMemory aggregation (M3 T3.3.2.1, T3.3.2.2, T3.3.2.3).

Pins the rollup contract:
- Papers: count + by-year + top tags / authors / venues.
- Ideas: count + by-status + avg critic score + most-engaged +
  top-scored.
- Discussions: session/message counts + role distribution + most
  recent sessions.
- ``since_days`` filters by ``created_at`` for all three tables.
- ``to_markdown`` produces a stable, predictable report layout.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from research_agent.agents.meta_memory import InsightsReport, MetaMemory
from research_agent.core.idea import IdeaStatus
from research_agent.core.paper import Paper
from research_agent.storage.database import Database, PaperRepository
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository


def _make_db(tmp_path) -> Database:
    return Database(tmp_path / "memory.db")


def _make_paper(pid: str, **kw) -> Paper:
    return Paper(
        id=pid,
        title=kw.get("title", f"Title for {pid}"),
        authors=kw.get("authors", ["A. Author"]),
        abstract=kw.get("abstract", "abs"),
        year=kw.get("year"),
        venue=kw.get("venue"),
        tags=kw.get("tags", []),
    )


# --------------------------------------------------------------------- empty


def test_empty_database_returns_zero_counts(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        meta = MetaMemory(db)
        report = meta.compute()
        assert report.paper_count == 0
        assert report.idea_count == 0
        assert report.session_count == 0
        assert report.message_count == 0
        assert report.period == "all-time"
        # Markdown still renders cleanly.
        md = report.to_markdown()
        assert "Research Insights" in md
        assert "Papers" in md
    finally:
        db.close()


# --------------------------------------------------------------------- papers


def test_papers_rollup(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        papers = PaperRepository(db)
        papers.save(
            _make_paper(
                "arxiv:1",
                year=2023,
                venue="NeurIPS",
                authors=["Alice", "Bob"],
                tags=["nlp", "transformers"],
            )
        )
        papers.save(
            _make_paper(
                "arxiv:2",
                year=2024,
                venue="ICML",
                authors=["Alice", "Carol"],
                tags=["nlp", "rlhf"],
            )
        )
        papers.save(
            _make_paper(
                "arxiv:3",
                year=2024,
                venue="ICML",
                authors=["Dave"],
                tags=["systems"],
            )
        )

        report = MetaMemory(db).compute()
        assert report.paper_count == 3
        assert dict(report.papers_by_year) == {2023: 1, 2024: 2}
        # Top tag: nlp (2) wins; transformers/rlhf/systems tie at 1.
        assert report.top_tags[0] == ("nlp", 2)
        # Top venue: ICML (2) > NeurIPS (1).
        assert report.top_venues[0] == ("ICML", 2)
        # Top author: Alice (2) > everyone else (1).
        assert report.top_authors[0] == ("Alice", 2)
    finally:
        db.close()


# ---------------------------------------------------------------------- ideas


def test_ideas_rollup(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        ideas = IdeaRepository(db)
        engaged = ideas.create("Engaged idea", "desc")
        ideas.append_score(engaged.id, 7.0, "good", session_id="s1")
        ideas.append_score(engaged.id, 8.0, "better", session_id="s2")
        ideas.append_score(engaged.id, 6.0, "regress", session_id="s3")

        top = ideas.create("Top scorer", "desc")
        ideas.append_score(top.id, 9.0, "great", session_id="s4")

        unshown = ideas.create("Just sitting", "desc")
        ideas.update_status(unshown.id, "shelved")

        report = MetaMemory(db).compute()
        assert report.idea_count == 3
        # current_score for engaged is the last append (6.0).
        # Average of (6.0, 9.0) = 7.5; "Just sitting" has critic_score=None
        # and is excluded.
        assert report.avg_critic_score == pytest.approx(7.5)
        # Top scorer comes first.
        assert report.top_scored_ideas[0][1] == "Top scorer"
        assert report.top_scored_ideas[0][2] == pytest.approx(9.0)
        # Most-debated is the engaged one (3 rounds).
        assert report.most_engaged_ideas[0][1] == "Engaged idea"
        assert report.most_engaged_ideas[0][2] == 3
        # Status mix: 2 active + 1 shelved (or whatever transitions allow).
        status_counts: dict[IdeaStatus, int] = report.ideas_by_status  # type: ignore[assignment]
        assert status_counts.get("active", 0) >= 2
        assert status_counts.get("shelved", 0) == 1
    finally:
        db.close()


def test_ideas_rollup_handles_no_scores(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        ideas = IdeaRepository(db)
        ideas.create("No score", "")
        report = MetaMemory(db).compute()
        assert report.avg_critic_score is None
        assert report.top_scored_ideas == []
        assert report.most_engaged_ideas == []
    finally:
        db.close()


# ---------------------------------------------------------------- discussions


def test_discussions_rollup(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        discussions = DiscussionRepository(db)
        discussions.append("sess-A", "user", "hello")
        discussions.append("sess-A", "analyst", "hi")
        discussions.append("sess-A", "critic", "but...")
        discussions.append("sess-B", "user", "topic 2")
        discussions.append("sess-B", "assistant", "reply")
        report = MetaMemory(db).compute()
        assert report.session_count == 2
        assert report.message_count == 5
        # role_counts must reflect the 5 inserts.
        assert sum(report.role_counts.values()) == 5
        # recent_session_ids includes both sessions.
        sids = [s for s, _ts in report.recent_session_ids]
        assert set(sids) == {"sess-A", "sess-B"}
    finally:
        db.close()


# --------------------------------------------------------------------- window


def test_since_days_filters_old_rows(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        # Old paper: backdate created_at by 60 days.
        old_ts = (
            datetime.now(tz=UTC) - timedelta(days=60)
        ).strftime("%Y-%m-%d %H:%M:%S")
        papers = PaperRepository(db)
        papers.save(_make_paper("arxiv:old", year=2022))
        papers.save(_make_paper("arxiv:new", year=2024))
        with db.conn:
            db.conn.execute(
                "UPDATE papers SET created_at = ? WHERE id = ?",
                (old_ts, "arxiv:old"),
            )

        all_report = MetaMemory(db).compute()
        assert all_report.paper_count == 2

        recent = MetaMemory(db).compute(since_days=30)
        assert recent.paper_count == 1
        assert recent.period == "last 30 days"
    finally:
        db.close()


# ------------------------------------------------------------------ markdown


def test_to_markdown_renders_known_sections(tmp_path) -> None:
    db = _make_db(tmp_path)
    try:
        papers = PaperRepository(db)
        papers.save(
            _make_paper("arxiv:m1", year=2024, venue="ICML", tags=["x"])
        )
        ideas = IdeaRepository(db)
        idea = ideas.create("idea m1", "")
        ideas.append_score(idea.id, 8.0, "r", session_id="s")
        DiscussionRepository(db).append("sess", "user", "hi")
        md = MetaMemory(db).compute().to_markdown()
        for header in (
            "# Research Insights",
            "## Papers",
            "## Ideas",
            "## Discussions",
        ):
            assert header in md
        # The Markdown shows our concrete data.
        assert "2024" in md
        assert "ICML" in md
        assert "idea m1" in md
    finally:
        db.close()


# ----------------------------------------------------- malformed defensive


def test_malformed_json_in_tags_is_tolerated(tmp_path) -> None:
    """Don't crash if tags column has a garbage value (e.g. legacy migration)."""
    db = _make_db(tmp_path)
    try:
        papers = PaperRepository(db)
        papers.save(_make_paper("arxiv:bad", year=2024))
        with db.conn:
            db.conn.execute(
                "UPDATE papers SET tags = ? WHERE id = ?",
                ("not-json", "arxiv:bad"),
            )
        report = MetaMemory(db).compute()
        # Just count; malformed tags fall through to no tag entries.
        assert report.paper_count == 1
        assert report.top_tags == []
    finally:
        db.close()


def test_report_is_a_dataclass() -> None:
    assert hasattr(InsightsReport, "__dataclass_fields__")
