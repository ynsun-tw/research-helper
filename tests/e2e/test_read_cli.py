"""E2E-style CLI tests for ``research read`` (mocked network + LLM)."""

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


def test_read_local_pdf_e2e(
    sample_pdf: Path,
    config_dir: Path,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()

    import research_agent.cli_services as svc
    from research_agent.core.llm import LLMClient, MockLLMProvider

    monkeypatch.setattr(svc, "load_paper", lambda source, cache_dir: paper)
    monkeypatch.setattr(
        LLMClient,
        "from_config",
        lambda config: MockLLMProvider([ANALYST_JSON, CRITIC_JSON]),
    )
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)

    result = runner.invoke(app, ["read", str(sample_pdf)])
    assert result.exit_code == 0, result.stdout
    assert "Analyst" in result.stdout
    assert "Critic" in result.stdout
    assert "Synthesis" in result.stdout

    db_path = config_dir / "memory.db"
    assert db_path.exists()


def test_discuss_exit_saves_session(
    config_dir: Path,
    paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    cfg.save()

    from research_agent.core.llm import LLMClient, MockLLMProvider

    mock = MockLLMProvider([IDEA_ANALYST_JSON, IDEA_CRITIC_JSON])
    monkeypatch.setattr(LLMClient, "from_config", lambda config: mock)
    monkeypatch.setattr("research_agent.cli._load_config", lambda: cfg)
    monkeypatch.setattr(
        "research_agent.cli_services._load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    result = runner.invoke(
        app,
        ["discuss", "--paper", "arxiv:1706.03762", "Apply attention to biology"],
        input="exit\n",
    )
    assert result.exit_code == 0, result.stdout
    assert "Session saved" in result.stdout
    assert "Analyst" in result.stdout
