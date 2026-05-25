"""Integration tests for Orchestrator dual-agent flow."""

from __future__ import annotations

import json

import pytest

from research_agent.agents.orchestrator import Orchestrator
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper, Section

ANALYST_JSON = json.dumps(
    {
        "contributions": ["Novel attention mechanism"],
        "method_insights": ["O(n^2) attention"],
        "potential_impact": "High",
        "related_work": ["RNN literature"],
        "claimed_vs_evidence": [],
        "confidence": 0.8,
    }
)

CRITIC_JSON = json.dumps(
    {
        "objections": ["Quadratic memory"],
        "support_score": 6,
        "score_reason": "Scaling concerns outweigh headline results",
        "honesty_note": "",
    }
)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer paper.",
        sections=[Section("Intro", "Attention replaces RNNs.")],
        full_text="Full transformer description.",
    )


@pytest.mark.asyncio
async def test_analyze_paper_parallel(paper: Paper) -> None:
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    orch = Orchestrator(llm)
    result = await orch.analyze_paper_parallel(paper)

    assert result.analyst.contributions
    assert result.critic.support_score == 6.0
    assert result.critic.score_reason
    assert "📄" not in result.summary  # summary is plain text
    assert result.summary


@pytest.mark.asyncio
async def test_route_read_paper_command(paper: Paper) -> None:
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    orch = Orchestrator(llm)
    task = orch.route("read_paper", {"paper": paper})
    assert task.command == "read_paper"
    out = await orch.run_task_async(task)
    assert hasattr(out, "analyst")
    assert hasattr(out, "conflicts")


def test_format_report_includes_both_perspectives(paper: Paper) -> None:
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    orch = Orchestrator(llm)
    agg = orch.run_task(orch.route("read", {"paper": paper}))
    report = agg.format_report()
    assert "Analyst" in report
    assert "Critic" in report
    assert "Synthesis" in report


@pytest.mark.asyncio
async def test_high_agreement_produces_consensus(paper: Paper) -> None:
    critic_high = json.dumps(
        {
            "objections": [],
            "support_score": 8,
            "score_reason": "Solid work",
            "honesty_note": "No major flaws",
        }
    )
    llm = MockLLMProvider([ANALYST_JSON, critic_high])
    result = await Orchestrator(llm).analyze_paper_parallel(paper)
    assert len(result.consensus) >= 1
