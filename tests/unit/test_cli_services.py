"""Unit tests for CLI service layer."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

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


def test_run_discuss_one_turn_and_exit(config_dir) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    inputs = iter(["hello", "exit"])

    code = run_discuss(cfg, llm, console, input_fn=lambda _: next(inputs))
    assert code == 0
    assert "Session saved" in console.file.getvalue()
