"""Unit tests for Idea debate schema and orchestration."""

from __future__ import annotations

import json

import pytest

from research_agent.agents.analyst import Analyst
from research_agent.agents.critic import Critic, parse_support_score
from research_agent.agents.debate import DebateHistory, DebateResult, FollowUpResult
from research_agent.agents.orchestrator import Orchestrator
from research_agent.core.llm import MockLLMProvider

IDEA_ANALYST_JSON = json.dumps(
    {
        "supports": ["Novel combination of X and Y"],
        "suggestions": ["Add human baseline"],
        "evidence": [{"claim": "Data exists", "basis": "Prior work in Z"}],
        "confidence": 0.7,
    }
)

FOLLOWUP_ANALYST_JSON = json.dumps({"conclusion": "The baseline concern is addressable with X."})
FOLLOWUP_CRITIC_JSON = json.dumps(
    {"conclusion": "You still need a clearer evaluation protocol."}
)

IDEA_CRITIC_JSON = json.dumps(
    {
        "objections": ["Evaluation protocol unclear"],
        "support_score": 6,
        "score_reason": "Needs stronger empirical plan",
        "suggestions": ["Pre-register benchmarks"],
        "honesty_note": "",
    }
)


def test_debate_result_from_agents() -> None:
    from research_agent.agents.analyst import IdeaSupportResult
    from research_agent.agents.critic import CritiqueResult

    analyst = IdeaSupportResult(supports=["A"], suggestions=["B"], evidence=[])
    critic = CritiqueResult(objections=["C"], support_score=6.0, score_reason="why")
    result = DebateResult.from_agent_results(analyst, critic, round_index=1, score_delta=-1.0)
    assert result.supports == ["A"]
    assert result.objections == ["C"]
    assert result.score == 6.0
    assert result.suggestions == ["B"]
    assert result.score_delta == -1.0


def test_analyst_analyze_idea() -> None:
    result = Analyst(MockLLMProvider([IDEA_ANALYST_JSON])).analyze_idea("My idea")
    assert result.supports
    assert result.confidence == 0.7


def test_critic_critique_idea() -> None:
    result = Critic(MockLLMProvider([IDEA_CRITIC_JSON])).critique_idea("My idea")
    assert result.support_score == 6.0
    assert result.suggestions


def test_parse_score_slash_ten() -> None:
    assert parse_support_score("The idea merits 7/10 with caveats") == 7.0
    assert parse_support_score("Rating: 8 out of 10") == 8.0


@pytest.mark.asyncio
async def test_debate_round_parallel() -> None:
    llm = MockLLMProvider([IDEA_ANALYST_JSON, IDEA_CRITIC_JSON])
    result = await Orchestrator(llm).debate_round_async("Test idea")
    assert result.supports
    assert result.objections
    assert 1.0 <= result.score <= 9.0


@pytest.mark.asyncio
async def test_debate_history_score_delta() -> None:
    llm = MockLLMProvider([IDEA_ANALYST_JSON, IDEA_CRITIC_JSON])
    orch = Orchestrator(llm)
    history = DebateHistory()
    r1 = await orch.debate_round_async("idea", history=history)
    history.append("round1", r1)
    critic_high = json.dumps(
        {
            "objections": [],
            "support_score": 8,
            "score_reason": "Improved",
            "suggestions": [],
            "honesty_note": "",
        }
    )
    llm.enqueue(IDEA_ANALYST_JSON)
    llm.enqueue(critic_high)
    r2 = await orch.debate_round_async("idea", history=history)
    assert r2.score_delta == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_followup_turn_returns_prose() -> None:
    llm = MockLLMProvider([FOLLOWUP_ANALYST_JSON, FOLLOWUP_CRITIC_JSON])
    orch = Orchestrator(llm)
    result = await orch.followup_turn_async(
        "My idea",
        "What about the baseline?",
        "Prior opening round context",
    )
    assert "baseline" in result.analyst_conclusion.lower()
    assert "evaluation" in result.critic_conclusion.lower()


def test_debate_history_tracks_followup_round() -> None:
    history = DebateHistory()
    history.append_followup(
        "follow-up q",
        FollowUpResult(analyst_conclusion="A", critic_conclusion="C"),
    )
    assert history.has_initial_round is False
    assert len(history.rounds) == 1
    assert history.rounds[0].followup is not None
