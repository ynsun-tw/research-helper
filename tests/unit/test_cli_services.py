"""Unit tests for CLI service layer."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

import research_agent.cli_services as svc
from research_agent.cli_services import run_discuss, run_read
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper

ANALYST_JSON = json.dumps(
    {
        "contributions": ["C1"],
        "method_insights": ["M1"],
        "potential_impact": "Impact",
        "related_work": ["R1"],
        "claimed_vs_evidence": [],
        "confidence": 0.8,
    }
)
CRITIC_JSON = json.dumps(
    {
        "objections": ["O1"],
        "support_score": 8,
        "score_reason": "Good",
        "honesty_note": "",
    }
)

IDEA_ANALYST_JSON = json.dumps(
    {
        "supports": ["Solid premise"],
        "suggestions": ["More baselines"],
        "evidence": [],
        "confidence": 0.8,
    }
)
IDEA_CRITIC_JSON = json.dumps(
    {
        "objections": ["Risky assumption"],
        "support_score": 7,
        "score_reason": "Viable with fixes",
        "suggestions": [],
        "honesty_note": "",
    }
)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="local:test",
        title="Test Paper",
        abstract="Abstract text.",
        full_text="Body.",
    )


def test_run_read_success(
    config_dir,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])

    import research_agent.cli_services as svc

    monkeypatch.setattr(svc, "load_paper", lambda source, cache_dir: paper)
    code = run_read("paper.pdf", cfg, llm, console)
    assert code == 0
    out = console.file.getvalue()
    assert "Analyst" in out
    assert "Critic" in out


def test_run_discuss_one_turn_and_exit(
    config_dir,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([IDEA_ANALYST_JSON, IDEA_CRITIC_JSON])
    inputs = iter(["hello", "exit"])

    monkeypatch.setattr(
        svc,
        "_load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    code = run_discuss(
        cfg,
        llm,
        console,
        paper_query="arxiv:1706.03762",
        input_fn=lambda _: next(inputs),
        use_chroma=False,
        prompt_save_idea=False,
    )
    assert code == 0
    out = console.file.getvalue()
    assert "Session saved" in out
    assert "message(s)" in out

    from research_agent.storage.database import Database
    from research_agent.storage.discussions import DiscussionRepository

    db = Database(cfg.db_path)
    repo = DiscussionRepository(db)
    row = db.conn.execute("SELECT session_id FROM discussions LIMIT 1").fetchone()
    assert row is not None
    messages = repo.list_session(row["session_id"])
    assert len(messages) == 3  # user + analyst + critic
    db.close()
