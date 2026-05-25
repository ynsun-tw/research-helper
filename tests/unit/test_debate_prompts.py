"""Paper-anchored debate prompt building."""

from __future__ import annotations

from research_agent.core.debate_prompts import idea_debate_user_prompt
from research_agent.core.paper import Paper


def test_idea_prompt_includes_anchor_paper() -> None:
    paper = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="Transformer architecture.",
        full_text="Self-attention replaces recurrence.",
    )
    prompt = idea_debate_user_prompt("Extend to biology", "", paper=paper)
    assert "ANCHOR PAPER" in prompt
    assert "Attention Is All You Need" in prompt
    assert "Extend to biology" in prompt
