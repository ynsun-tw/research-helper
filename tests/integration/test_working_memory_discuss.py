"""Integration: WorkingMemory drives Orchestrator discuss turns."""

from __future__ import annotations

import json

import pytest

from research_agent.agents.orchestrator import Orchestrator
from research_agent.core.llm import MockLLMProvider
from research_agent.memory.working_memory import WorkingMemory

ANALYST_JSON = json.dumps(
    {
        "contributions": ["Point"],
        "method_insights": ["Detail"],
        "potential_impact": "Medium",
        "related_work": [],
        "claimed_vs_evidence": [],
        "confidence": 0.7,
    }
)
CRITIC_JSON = json.dumps(
    {
        "objections": ["Gap"],
        "support_score": 6,
        "score_reason": "Needs work",
        "honesty_note": "",
    }
)


@pytest.mark.asyncio
async def test_discuss_turn_uses_memory_context() -> None:
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    orch = Orchestrator(llm)
    mem = WorkingMemory.new_session()
    mem.append("user", "First question about transformers")
    analyst, critic = await orch.discuss_turn_async(mem, max_context_tokens=4000)
    assert analyst.role == "analyst"
    assert critic.role == "critic"
    assert llm.calls
    user_msg = llm.calls[0][1].content
    assert "First question" in user_msg


@pytest.mark.asyncio
async def test_discuss_truncates_long_history_in_prompt() -> None:
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    orch = Orchestrator(llm)
    mem = WorkingMemory.new_session()
    mem.append("user", "OLD " * 500)
    mem.append("analyst", "ack")
    mem.append("user", "CURRENT")
    await orch.discuss_turn_async(mem, max_context_tokens=50)
    prompt = llm.calls[0][1].content
    assert "CURRENT" in prompt
    assert "OLD " * 20 not in prompt


@pytest.mark.asyncio
async def test_route_discuss_with_memory() -> None:
    llm = MockLLMProvider([ANALYST_JSON, CRITIC_JSON])
    mem = WorkingMemory.new_session()
    mem.append("user", "idea")
    task = Orchestrator(llm).route("discuss", {"memory": mem})
    out = await Orchestrator(llm).run_task_async(task)
    assert out["analyst"].role == "analyst"
