"""Performance baseline + DB index regression tests for M5 S5.4.1.

These are NOT micro-benchmarks. They:

1. Assert that the indexes the schema promises actually exist after
   ``Database.__init__`` so a refactor can't silently drop one and
   regress query plans on production-scale data.
2. Hold a wall-clock budget around a representative DB query at
   1000-row scale (the acceptance criterion is ``< 2 s`` for vector
   retrieval; SQLite reads at this scale should be orders of
   magnitude faster — we check ``< 500 ms`` as a sanity ceiling that
   trips long before users notice).
3. Confirm that the SQLite query planner actually picks one of the
   new indexes for the canonical "filter by session" /
   "filter by status" queries (``EXPLAIN QUERY PLAN`` inspection).

CLI startup time is intentionally NOT timed in this test file —
import latency varies wildly by hardware and would just flake CI.
The lazy-import refactor is exercised indirectly by the existing
CLI tests; documentation captures the measurement methodology.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from research_agent.storage.database import Database

# --- index regression -------------------------------------------------------


def _index_names(db: Database, table: str) -> set[str]:
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table,),
    ).fetchall()
    return {row[0] for row in rows if row[0]}


def test_papers_has_expected_indexes(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        names = _index_names(db, "papers")
    finally:
        db.close()
    assert "idx_papers_title" in names
    assert "idx_papers_created" in names


def test_discussions_has_session_created_composite(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        names = _index_names(db, "discussions")
    finally:
        db.close()
    assert "idx_discussions_session" in names
    assert "idx_discussions_session_created" in names
    assert "idx_discussions_created" in names


def test_ideas_has_updated_status_created_indexes(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        names = _index_names(db, "ideas")
    finally:
        db.close()
    assert "idx_ideas_updated" in names
    assert "idx_ideas_status" in names
    assert "idx_ideas_created" in names


def test_style_samples_index_preserved(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        names = _index_names(db, "style_samples")
    finally:
        db.close()
    assert "idx_style_samples_paper" in names


# --- query plan inspection --------------------------------------------------


def _plan(db: Database, sql: str, params: tuple = ()) -> str:
    rows = db.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return "\n".join(" ".join(str(c) for c in row) for row in rows)


def test_query_plan_filter_by_session_uses_index(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        # Populate enough rows that SQLite's planner doesn't decide a
        # full scan is cheaper.
        for i in range(500):
            db.conn.execute(
                "INSERT INTO discussions (id, session_id, role, content) "
                "VALUES (?, ?, ?, ?)",
                (uuid.uuid4().hex, f"session-{i % 25}", "user", "x"),
            )
        db.conn.commit()
        plan = _plan(
            db,
            "SELECT * FROM discussions WHERE session_id = ? ORDER BY created_at",
            ("session-3",),
        )
    finally:
        db.close()
    # We just want the planner to use *some* index on discussions,
    # not the literal pathological full scan.
    assert "idx_discussions_session" in plan, plan


def test_query_plan_filter_by_status_uses_index(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        for i in range(500):
            db.conn.execute(
                "INSERT INTO ideas (id, title, status) VALUES (?, ?, ?)",
                (
                    uuid.uuid4().hex,
                    f"idea {i}",
                    ("active", "shelved", "waiting")[i % 3],
                ),
            )
        db.conn.commit()
        plan = _plan(db, "SELECT id FROM ideas WHERE status = ?", ("shelved",))
    finally:
        db.close()
    assert "idx_ideas_status" in plan, plan


# --- wall-clock sanity ceiling ----------------------------------------------


def test_papers_list_under_500ms_at_1k_rows(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        with db.conn:
            for i in range(1000):
                db.conn.execute(
                    "INSERT INTO papers (id, title, abstract) VALUES (?, ?, ?)",
                    (f"p{i:04d}", f"Title {i}", "abstract " * 20),
                )
        # Cold-cache pass first, then a warm pass; we time the warm
        # pass because that's what the user sees in steady state.
        db.conn.execute(
            "SELECT id, title FROM papers ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        t0 = time.perf_counter()
        rows = db.conn.execute(
            "SELECT id, title FROM papers ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        elapsed = time.perf_counter() - t0
    finally:
        db.close()
    assert len(rows) == 50
    # 500 ms is a wide ceiling intentionally — real timings are sub-ms
    # locally. We just want a tripwire if a future change makes a
    # 1k-row query unusable.
    assert elapsed < 0.5, f"query took {elapsed * 1000:.1f} ms"


def test_discussions_session_query_under_500ms_at_1k_rows(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    try:
        with db.conn:
            for i in range(1000):
                db.conn.execute(
                    "INSERT INTO discussions (id, session_id, role, content) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        uuid.uuid4().hex,
                        f"session-{i % 50}",
                        "user",
                        "content " * 5,
                    ),
                )
        db.conn.execute(
            "SELECT id, content FROM discussions WHERE session_id = ? "
            "ORDER BY created_at LIMIT 50",
            ("session-3",),
        ).fetchall()
        t0 = time.perf_counter()
        db.conn.execute(
            "SELECT id, content FROM discussions WHERE session_id = ? "
            "ORDER BY created_at LIMIT 50",
            ("session-3",),
        ).fetchall()
        elapsed = time.perf_counter() - t0
    finally:
        db.close()
    assert elapsed < 0.5, f"query took {elapsed * 1000:.1f} ms"


# --- migration idempotence --------------------------------------------------


def test_reopening_db_does_not_duplicate_indexes(tmp_path: Path) -> None:
    """``CREATE INDEX IF NOT EXISTS`` must be idempotent across opens."""
    path = tmp_path / "x.db"
    Database(path).close()
    db = Database(path)
    try:
        # No duplicate indexes: each name appears exactly once.
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%'"
        ).fetchall()
    finally:
        db.close()
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)), names


# --- cli.DEFAULT_CHECK_THRESHOLD parity with plagiarism module --------------


def test_check_threshold_default_matches_plagiarism_module() -> None:
    """``cli.DEFAULT_CHECK_THRESHOLD`` must mirror ``plagiarism.DEFAULT_THRESHOLD``.

    The CLI keeps the threshold as a literal to avoid pulling in the
    plagiarism tokenizer at import time; this test prevents the two
    constants from drifting.
    """
    from research_agent.cli import DEFAULT_CHECK_THRESHOLD
    from research_agent.style.plagiarism import DEFAULT_THRESHOLD

    assert DEFAULT_CHECK_THRESHOLD == DEFAULT_THRESHOLD


# --- raw sqlite sanity (no Database wrapper) --------------------------------


def test_schema_creates_no_warnings(tmp_path: Path) -> None:
    """Schema should be valid SQL — sqlite raising an error here would
    indicate a typo no test ever exercised. Belt and braces."""
    db = Database(tmp_path / "x.db")
    try:
        # Touch every table once to confirm they all exist.
        for table in (
            "papers",
            "ideas",
            "discussions",
            "search_queries",
            "search_results",
            "reading_queue",
            "style_samples",
            "draft_revisions",
        ):
            cur = db.conn.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 0
    except sqlite3.Error as exc:  # pragma: no cover - schema would be broken
        raise AssertionError(f"schema is invalid: {exc}") from exc
    finally:
        db.close()
