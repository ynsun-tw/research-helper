"""E2E: batch-read 3 papers through the queue (M3 T3.2.2.4).

End-to-end verification of the /queue add → /queue read → auto-mark-done
loop the user runs to clear their reading backlog. Unit-level tests in
tests/unit/test_reading_queue.py cover the repo + slash command + FIFO
semantics in isolation; this test stitches them together with the real
REPL driver and the real SQLite schema.

Network is mocked (no arXiv hit, no S2 hit). The LLM is mocked
(Analyst + Critic JSON pairs replayed by MockLLMProvider). What we
exercise for real:
- REPL slash dispatch
- ChatSession lifecycle (open, persist, close, vector index hook)
- ReadingQueueRepository SQLite I/O
- PaperRepository SQLite I/O (paper-saved-after-analysis)
- Auto-mark-done hook in _load_and_analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Config
from research_agent.core.paper import Paper, Section

runner = CliRunner()


def _analyst_json(contributions: str) -> str:
    return json.dumps(
        {
            "contributions": [contributions],
            "method_insights": ["m"],
            "potential_impact": "ok",
            "related_work": ["r"],
            "claimed_vs_evidence": [],
            "confidence": 0.8,
        }
    )


def _critic_json(score: int) -> str:
    return json.dumps(
        {
            "objections": ["o"],
            "support_score": score,
            "score_reason": "test",
            "honesty_note": "",
        }
    )


PAPERS = [
    Paper(
        id=f"arxiv:{aid}",
        title=title,
        abstract=f"Abstract for {title}.",
        sections=[Section("Intro", "...")],
        full_text=f"Full text of {title}.",
    )
    for aid, title in (
        ("1706.03762", "Attention Is All You Need"),
        ("1810.04805", "BERT"),
        ("2005.14165", "GPT-3"),
    )
]


def test_queue_batch_read_three_papers_end_to_end(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()

    from research_agent.core.llm import LLMClient, MockLLMProvider

    # 3 papers x 2 LLM calls (Analyst + Critic) = 6 responses.
    responses = [
        _analyst_json("A1"), _critic_json(7),
        _analyst_json("A2"), _critic_json(8),
        _analyst_json("A3"), _critic_json(6),
    ]
    mock = MockLLMProvider(responses)
    monkeypatch.setattr(LLMClient, "from_config", lambda config: mock)
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)

    # _load_anchor_paper would otherwise touch the network. Return the
    # right Paper for each arxiv id the slash command was invoked with.
    by_id = {p.id.removeprefix("arxiv:"): p for p in PAPERS}

    def fake_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        # First positional arg is the user-supplied source string.
        source = args[0] if args else kwargs.get("source", "")
        return by_id[str(source).strip()]

    monkeypatch.setattr(
        "research_agent.chat.tools._load_anchor_paper",
        fake_loader,
    )

    script = (
        "/queue add 1706.03762 Attention Is All You Need\n"
        "/queue add 1810.04805 BERT\n"
        "/queue add 2005.14165 GPT-3\n"
        "/queue list\n"
        "/queue read\n"
        "/queue read\n"
        "/queue read\n"
        "/queue list all\n"
        "/exit\n"
    )

    result = runner.invoke(app, [], input=script)
    assert result.exit_code == 0, result.stdout

    # All three titles surfaced via the queue list at least once.
    for paper in PAPERS:
        assert paper.title in result.stdout, (
            f"{paper.title} missing from session output"
        )

    # All three Analyst/Critic reports rendered.
    assert result.stdout.count("Analyst") >= 3
    assert result.stdout.count("Critic") >= 3

    # Final '/queue list all' must show each paper as done.
    tail = result.stdout.split("/queue list all", 1)[-1]
    assert tail.lower().count("done") >= 3, (
        f"Expected >= 3 'done' markers in tail of output; got:\n{tail}"
    )

    # Hit the SQLite store directly to assert state, not just rendering.
    from research_agent.storage.database import Database, PaperRepository
    from research_agent.storage.reading_queue import ReadingQueueRepository

    db = Database(cfg.db_path)
    try:
        queue_repo = ReadingQueueRepository(db)
        entries = queue_repo.list()
        statuses = {e.arxiv_id: e.status for e in entries}
        assert statuses == {
            "1706.03762": "done",
            "1810.04805": "done",
            "2005.14165": "done",
        }
        assert queue_repo.next_pending() is None, (
            "Queue should be empty of pending entries after 3 reads"
        )

        # All three papers got persisted to the library (PaperRepository).
        paper_repo = PaperRepository(db)
        for paper in PAPERS:
            stored = paper_repo.get(paper.id)
            assert stored is not None, f"{paper.id} not in PaperRepository"
            assert stored.title == paper.title
    finally:
        db.close()


def test_queue_batch_read_handles_empty_queue_after_drain(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After draining the queue, an extra /queue read must say so cleanly
    rather than crashing or re-loading the last paper."""
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()

    from research_agent.core.llm import LLMClient, MockLLMProvider

    responses = [_analyst_json("only"), _critic_json(7)]
    monkeypatch.setattr(
        LLMClient,
        "from_config",
        lambda config: MockLLMProvider(responses),
    )
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)
    monkeypatch.setattr(
        "research_agent.chat.tools._load_anchor_paper",
        lambda *args, **kwargs: PAPERS[0],
    )

    script = (
        "/queue add 1706.03762 AIAYN\n"
        "/queue read\n"
        "/queue read\n"  # second read - queue is empty
        "/exit\n"
    )
    result = runner.invoke(app, [], input=script)
    assert result.exit_code == 0, result.stdout
    assert "queue" in result.stdout.lower()
    # The second /queue read should communicate emptiness, not crash.
    lower = result.stdout.lower()
    assert any(token in lower for token in ("empty", "no pending", "nothing"))
