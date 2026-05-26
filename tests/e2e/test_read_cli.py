"""E2E-style CLI tests for the conversational shell (mocked network + LLM).

After M2.5 the `research` command is a single REPL. Paper loading and idea
debate happen through slash commands inside the chat shell.
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

ANALYST_JSON = json.dumps(
    {
        "contributions": ["Main idea"],
        "method_insights": ["Uses attention"],
        "potential_impact": "High",
        "related_work": ["Prior RNN work"],
        "claimed_vs_evidence": [],
        "confidence": 0.75,
    }
)

CRITIC_JSON = json.dumps(
    {
        "objections": ["Compute cost"],
        "support_score": 7,
        "score_reason": "Solid but expensive",
        "honesty_note": "",
    }
)

IDEA_ANALYST_JSON = json.dumps(
    {
        "supports": ["Applies attention well"],
        "suggestions": ["Add baseline"],
        "evidence": [],
        "confidence": 0.7,
    }
)
IDEA_CRITIC_JSON = json.dumps(
    {
        "objections": ["Dataset unclear"],
        "support_score": 6,
        "score_reason": "Needs evaluation plan",
        "suggestions": [],
        "honesty_note": "",
    }
)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer architecture.",
        sections=[Section("Intro", "Self-attention.")],
        full_text="Transformer details.",
    )


def test_repl_slash_read_runs_analysis(
    config_dir: Path,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()

    from research_agent.core.llm import LLMClient, MockLLMProvider

    monkeypatch.setattr(
        LLMClient,
        "from_config",
        lambda config: MockLLMProvider([ANALYST_JSON, CRITIC_JSON]),
    )
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)
    monkeypatch.setattr(
        "research_agent.chat.tools._load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    result = runner.invoke(app, [], input="/read arxiv:1706.03762\n/exit\n")
    assert result.exit_code == 0, result.stdout
    assert "Analyst" in result.stdout
    assert "Critic" in result.stdout
    assert "Synthesis" in result.stdout

    db_path = config_dir / "memory.db"
    assert db_path.exists()


def test_repl_slash_discuss_saves_session(
    config_dir: Path,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()

    from research_agent.core.llm import LLMClient, MockLLMProvider

    # First two responses fuel /read (Analyst + Critic),
    # next two fuel /discuss (idea Analyst + idea Critic).
    mock = MockLLMProvider(
        [ANALYST_JSON, CRITIC_JSON, IDEA_ANALYST_JSON, IDEA_CRITIC_JSON]
    )
    monkeypatch.setattr(LLMClient, "from_config", lambda config: mock)
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)
    monkeypatch.setattr(
        "research_agent.chat.tools._load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    result = runner.invoke(
        app,
        [],
        input=(
            "/read arxiv:1706.03762\n"
            "/discuss Apply attention to biology\n"
            "/exit\n"
        ),
    )
    assert result.exit_code == 0, result.stdout
    assert "Session saved" in result.stdout
    assert "Analyst" in result.stdout
