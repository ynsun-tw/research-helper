"""Unit tests for Analyst, Critic, and prompt loading."""

from __future__ import annotations

import json

import pytest

from research_agent.agents.analyst import Analyst
from research_agent.agents.critic import Critic, normalize_score, parse_support_score
from research_agent.agents.prompts import load_system_prompt
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper, Section


def _sample_paper() -> Paper:
    return Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        abstract="We propose the Transformer architecture.",
        sections=[Section(title="1. Introduction", content="Recurrent models are slow.")],
        full_text="Transformer uses self-attention.",
    )


ANALYST_JSON = json.dumps(
    {
        "contributions": ["Introduces pure attention architecture"],
        "method_insights": ["Self-attention replaces recurrence"],
        "potential_impact": "Foundation for modern LLMs",
        "related_work": ["Bahdanau et al. 2014"],
        "claimed_vs_evidence": [
            {"claim": "State-of-the-art on WMT", "evidence": "BLEU gains reported in Table 2"}
        ],
        "confidence": 0.85,
    }
)

CRITIC_JSON = json.dumps(
    {
        "objections": ["Limited analysis of compute cost at scale"],
        "support_score": 8,
        "score_reason": "Strong empirical results with minor scalability concerns",
        "honesty_note": "",
    }
)


def test_load_analyst_prompt() -> None:
    prompt = load_system_prompt("analyst")
    assert "Analyst" in prompt
    assert "JSON" in prompt


def test_load_critic_prompt_forbids_ten() -> None:
    prompt = load_system_prompt("critic")
    assert "NEVER give 10" in prompt or "never" in prompt.lower()


def test_analyst_analyze_paper() -> None:
    llm = MockLLMProvider([ANALYST_JSON])
    result = Analyst(llm).analyze_paper(_sample_paper())
    assert "attention" in result.contributions[0].lower()
    assert result.confidence == 0.85
    assert len(result.claimed_vs_evidence) == 1


def test_critic_critique_paper() -> None:
    llm = MockLLMProvider([CRITIC_JSON])
    result = Critic(llm).critique_paper(_sample_paper())
    assert result.support_score == 8.0
    assert len(result.objections) >= 1


def test_critic_score_never_ten() -> None:
    raw = json.dumps({"objections": ["x"], "support_score": 10, "score_reason": "perfect"})
    result = Critic(MockLLMProvider([raw])).critique_paper(_sample_paper())
    assert result.support_score <= 9.0


def test_critic_low_score_requires_reason() -> None:
    raw = json.dumps({"objections": ["weak baselines"], "support_score": 5, "score_reason": ""})
    result = Critic(MockLLMProvider([raw])).critique_paper(_sample_paper())
    assert result.support_score == 5.0
    assert result.score_reason


def test_parse_support_score_from_text() -> None:
    assert parse_support_score('{"support_score": 7}') == 7.0
    assert parse_support_score("Overall support score: 6/9") == 6.0
    assert parse_support_score("Overall 7/10 with caveats") == 7.0


def test_normalize_score_clamps() -> None:
    assert normalize_score(10) == 9.0
    assert normalize_score(0) == 1.0
    assert normalize_score("bad") == 5.0


def test_analyst_run_returns_agent_response() -> None:
    llm = MockLLMProvider([ANALYST_JSON])
    resp = Analyst(llm).run({"paper": _sample_paper()})
    assert resp.role == "analyst"
    assert resp.confidence == 0.85


def test_critic_run_missing_paper_raises() -> None:
    with pytest.raises(TypeError):
        Critic(MockLLMProvider()).run({})


IDEA_ANALYST_JSON = json.dumps(
    {
        "supports": ["Concrete benefit"],
        "suggestions": ["Pilot study"],
        "evidence": [{"claim": "Assumption A", "basis": "Prior art"}],
        "confidence": 0.6,
    }
)
IDEA_CRITIC_JSON = json.dumps(
    {
        "objections": ["Scope too broad"],
        "support_score": 5,
        "score_reason": "Under-specified evaluation",
        "suggestions": ["Narrow task"],
        "honesty_note": "",
    }
)


def test_load_analyst_idea_prompt() -> None:
    prompt = load_system_prompt("analyst_idea")
    assert "Idea debate" in prompt or "idea" in prompt.lower()


def test_analyst_analyze_idea() -> None:
    result = Analyst(MockLLMProvider([IDEA_ANALYST_JSON])).analyze_idea("New method")
    assert result.supports
    assert result.confidence == 0.6


def test_critic_critique_idea() -> None:
    result = Critic(MockLLMProvider([IDEA_CRITIC_JSON])).critique_idea("New method")
    assert result.support_score == 5.0
    assert result.score_reason
