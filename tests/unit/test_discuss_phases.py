"""Discuss: structured opening round vs prose follow-ups."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

import research_agent.cli_services as svc
from research_agent.cli_services import run_discuss
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper

IDEA_ANALYST_JSON = json.dumps(
    {
        "supports": ["S1"],
        "suggestions": ["G1"],
        "evidence": [],
        "confidence": 0.7,
    }
)
IDEA_CRITIC_JSON = json.dumps(
    {
        "objections": ["O1"],
        "support_score": 6,
        "score_reason": "Needs work",
        "suggestions": [],
        "honesty_note": "",
    }
)
FOLLOWUP_ANALYST_JSON = json.dumps({"conclusion": "Analyst follow-up answer."})
FOLLOWUP_CRITIC_JSON = json.dumps({"conclusion": "Critic follow-up answer."})


@pytest.fixture
def anchor_paper() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer paper.",
        full_text="Self-attention mechanism.",
    )


def test_discuss_second_turn_uses_followup_format(
    config_dir,
    anchor_paper: Paper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    llm = MockLLMProvider(
        [
            IDEA_ANALYST_JSON,
            IDEA_CRITIC_JSON,
            FOLLOWUP_ANALYST_JSON,
            FOLLOWUP_CRITIC_JSON,
        ]
    )
    inputs = iter(["my idea", "what about baselines?", "exit"])

    monkeypatch.setattr(
        svc,
        "_load_anchor_paper",
        lambda *args, **kwargs: anchor_paper,
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
    assert "Attention Is All You Need" in out
    assert "6/9" in out
    assert "- S1" in out
    assert "Analyst follow-up answer." in out
    assert "Critic follow-up answer." in out
    assert out.count("6/9") == 1
