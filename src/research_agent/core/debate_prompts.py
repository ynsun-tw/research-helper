"""Shared user-message blocks for paper-anchored idea debate."""

from __future__ import annotations

from research_agent.core.paper import Paper
from research_agent.core.paper_context import format_paper_for_prompt


def anchor_paper_section(paper: Paper | None) -> str:
    if paper is None:
        return ""
    return (
        "ANCHOR PAPER — ground ALL reasoning in this paper only. "
        "Do not speculate beyond its claims or invent citations.\n\n"
        f"{format_paper_for_prompt(paper)}\n"
    )


def idea_debate_user_prompt(
    idea_text: str,
    context: str,
    *,
    paper: Paper | None,
    user_message: str | None = None,
) -> str:
    parts: list[str] = []
    anchor = anchor_paper_section(paper)
    if anchor:
        parts.append(anchor)
    if user_message is None:
        parts.append(f"User research idea (about the anchor paper):\n\n{idea_text}")
    else:
        parts.append(f"Original idea (about the anchor paper):\n\n{idea_text}")
        parts.append(f"\nUser follow-up:\n\n{user_message}")
    if context.strip():
        parts.append(f"\nPrior debate context:\n{context}")
    return "\n".join(parts)
