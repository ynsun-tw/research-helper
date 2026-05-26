"""Critic agent — devil's advocate with scored support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent, extract_json
from research_agent.agents.writing_pipeline import (
    WritingReview,
    build_review_prompt,
    parse_review,
)
from research_agent.core.debate_prompts import idea_debate_user_prompt
from research_agent.core.paper import Paper


@dataclass(slots=True)
class CritiqueResult:
    objections: list[str]
    support_score: float
    score_reason: str = ""
    honesty_note: str = ""
    suggestions: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_agent_response(self) -> AgentResponse:
        lines = [
            f"## Support score: {self.support_score:.0f}/9",
            f"**Reason:** {self.score_reason or '(not provided)'}",
            "",
            "## Objections",
        ]
        if self.objections:
            lines.extend(f"- {o}" for o in self.objections)
        elif self.honesty_note:
            lines.append(f"- {self.honesty_note}")
        else:
            lines.append("- No major objections identified.")
        return AgentResponse(
            role="critic",
            content="\n".join(lines),
            score=self.support_score,
            metadata={"critique": self},
        )


class Critic(BaseAgent):
    role = "critic"

    @property
    def prompt_name(self) -> str:
        return "critic"

    def run(self, context: dict[str, Any]) -> AgentResponse:
        paper = context.get("paper")
        if not isinstance(paper, Paper):
            raise TypeError("context['paper'] must be a Paper instance")
        result = self.critique_paper(paper)
        return result.to_agent_response()

    def critique_paper(self, paper: Paper) -> CritiqueResult:
        prompt = _paper_prompt(paper)
        raw = self._chat(prompt, temperature=0.3)
        return _parse_critique(raw)

    def critique_idea(
        self,
        idea_text: str,
        context: str = "",
        *,
        paper: Paper | None = None,
    ) -> CritiqueResult:
        prompt = idea_debate_user_prompt(idea_text, context, paper=paper)
        raw = self._chat(prompt, temperature=0.3)
        return _parse_critique(raw)

    def followup_idea(
        self,
        idea_text: str,
        user_message: str,
        context: str = "",
        *,
        paper: Paper | None = None,
    ) -> str:
        prompt = idea_debate_user_prompt(
            idea_text, context, paper=paper, user_message=user_message
        )
        raw = self._chat(prompt, temperature=0.3)
        return _parse_followup_conclusion(raw)

    def review_writing(self, draft_text: str, section: str) -> WritingReview:
        """Review a Scribe draft for overclaim / unsupported conclusions.

        Caller is expected to instantiate this Critic with
        ``prompt_stem="critic_writing"`` so the system prompt is the
        writing-review variant (see
        :mod:`research_agent.agents.writing_pipeline`).
        """
        prompt = build_review_prompt(draft_text=draft_text, section=section)
        raw = self._chat(prompt, temperature=0.3)
        return parse_review("critic", raw)


def _paper_prompt(paper: Paper) -> str:
    sections = "\n\n".join(f"### {s.title}\n{s.content[:2000]}" for s in paper.sections[:8])
    body = paper.full_text[:12000] if paper.full_text else sections
    return (
        f"Critique this paper.\n\n"
        f"Title: {paper.title}\n"
        f"ID: {paper.id}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        f"Sections:\n{sections or '(none detected)'}\n\n"
        f"Full text excerpt:\n{body}"
    )


def parse_support_score(text: str, parsed: dict[str, object] | None = None) -> float:
    """Extract support score from JSON or free text; never return 10."""
    if parsed is not None:
        for key in ("support_score", "score", "rating"):
            if key in parsed:
                return normalize_score(parsed[key])
    patterns = (
        r'"support_score"\s*:\s*(\d+(?:\.\d+)?)',
        r"support[_ ]?score[:\s]+(\d+(?:\.\d+)?)",
        r"score[:\s]+(\d+(?:\.\d+)?)\s*/\s*(?:9|10)",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s+out\s+of\s+10",
        r"score[:\s]+(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_score(m.group(1))
    return 5.0


def normalize_score(value: object) -> float:
    """Clamp to 1-9 (10 is forbidden by product rules)."""
    try:
        if isinstance(value, (int, float, str)):
            score = float(value)
        else:
            return 5.0
    except ValueError:
        return 5.0
    if score >= 10:
        score = 9.0
    return max(1.0, min(9.0, score))


def _parse_critique(raw: str) -> CritiqueResult:
    data = extract_json(raw)
    objections = _as_str_list(data.get("objections"))
    score = parse_support_score(raw, data)
    score_reason = str(data.get("score_reason", "")).strip()
    honesty_note = str(data.get("honesty_note", "")).strip()

    if score < 7 and not score_reason:
        score_reason = "Score below 7 requires justification; model did not provide score_reason."
    if not objections and not honesty_note:
        honesty_note = "No substantive objections found; paper appears reasonably sound."

    return CritiqueResult(
        objections=objections,
        support_score=score,
        score_reason=score_reason,
        honesty_note=honesty_note,
        suggestions=_as_str_list(data.get("suggestions")),
        raw_response=raw,
    )


def _parse_followup_conclusion(raw: str) -> str:
    data = extract_json(raw)
    text = str(data.get("conclusion", "")).strip()
    if text:
        return text
    return raw.strip()


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v]
