"""Analyst agent — curiosity-driven paper understanding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent
from research_agent.agents.schemas import (
    AnalysisPayload,
    ConclusionPayload,
    IdeaSupportPayload,
    parse_model,
)
from research_agent.agents.writing_pipeline import (
    WritingReview,
    build_review_prompt,
    parse_review,
)
from research_agent.core.debate_prompts import idea_debate_user_prompt
from research_agent.core.paper import Paper


@dataclass(slots=True)
class ClaimedVsEvidence:
    claim: str
    evidence: str


@dataclass(slots=True)
class AnalysisResult:
    contributions: list[str]
    method_insights: list[str]
    potential_impact: str
    related_work: list[str]
    claimed_vs_evidence: list[ClaimedVsEvidence] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""

    def to_agent_response(self) -> AgentResponse:
        lines = [
            "## Core contributions",
            *[f"- {c}" for c in self.contributions],
            "",
            "## Method insights",
            *[f"- {m}" for m in self.method_insights],
            "",
            f"## Potential impact\n{self.potential_impact}",
            "",
            "## Related work",
            *[f"- {r}" for r in self.related_work],
            "",
            "## Claims vs evidence",
        ]
        for pair in self.claimed_vs_evidence:
            lines.append(f"- **Claim:** {pair.claim}")
            lines.append(f"  **Evidence:** {pair.evidence}")
        return AgentResponse(
            role="analyst",
            content="\n".join(lines),
            confidence=self.confidence,
            metadata={"analysis": self},
        )


class Analyst(BaseAgent):
    role = "analyst"

    @property
    def prompt_name(self) -> str:
        return "analyst"

    def run(self, context: dict[str, Any]) -> AgentResponse:
        paper = context.get("paper")
        if not isinstance(paper, Paper):
            raise TypeError("context['paper'] must be a Paper instance")
        result = self.analyze_paper(paper)
        return result.to_agent_response()

    def analyze_paper(self, paper: Paper) -> AnalysisResult:
        prompt = _paper_prompt(paper)
        raw = self._chat(prompt)
        return _parse_analysis(raw)

    def analyze_idea(
        self,
        idea_text: str,
        context: str = "",
        *,
        paper: Paper | None = None,
    ) -> IdeaSupportResult:
        prompt = idea_debate_user_prompt(idea_text, context, paper=paper)
        raw = self._chat(prompt)
        return _parse_idea_support(raw)

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
        raw = self._chat(prompt)
        return _parse_conclusion(raw)

    def review_writing(self, draft_text: str, section: str) -> WritingReview:
        """Review a Scribe draft for argument sufficiency / differentiation.

        Caller is expected to instantiate this Analyst with
        ``prompt_stem="analyst_writing"`` so the system prompt targets
        writing review (see :mod:`research_agent.agents.writing_pipeline`
        for the prompt format).
        """
        prompt = build_review_prompt(draft_text=draft_text, section=section)
        raw = self._chat(prompt, temperature=0.3)
        return parse_review("analyst", raw)


@dataclass(slots=True)
class IdeaSupportResult:
    supports: list[str]
    suggestions: list[str]
    evidence: list[ClaimedVsEvidence]
    confidence: float = 0.5
    raw_response: str = ""


def _parse_conclusion(raw: str) -> str:
    payload = parse_model(raw, ConclusionPayload)
    return payload.conclusion or raw.strip()


def _parse_idea_support(raw: str) -> IdeaSupportResult:
    payload = parse_model(raw, IdeaSupportPayload)
    evidence = [
        ClaimedVsEvidence(claim=item.claim, evidence=item.evidence)
        for item in payload.evidence
    ]
    return IdeaSupportResult(
        supports=payload.supports,
        suggestions=payload.suggestions,
        evidence=evidence,
        confidence=payload.confidence,
        raw_response=raw,
    )


def _paper_prompt(paper: Paper) -> str:
    sections = "\n\n".join(f"### {s.title}\n{s.content[:2000]}" for s in paper.sections[:8])
    body = paper.full_text[:12000] if paper.full_text else sections
    authors = ", ".join(paper.authors) if paper.authors else "Unknown"
    return (
        f"Analyze this paper.\n\n"
        f"Title: {paper.title}\n"
        f"Authors: {authors}\n"
        f"ID: {paper.id}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        f"Sections:\n{sections or '(none detected)'}\n\n"
        f"Full text excerpt:\n{body}"
    )


def _parse_analysis(raw: str) -> AnalysisResult:
    payload = parse_model(raw, AnalysisPayload)
    pairs = [
        ClaimedVsEvidence(claim=item.claim, evidence=item.evidence)
        for item in payload.claimed_vs_evidence
    ]
    return AnalysisResult(
        contributions=payload.contributions,
        method_insights=payload.method_insights,
        potential_impact=payload.potential_impact,
        related_work=payload.related_work,
        claimed_vs_evidence=pairs,
        confidence=payload.confidence,
        raw_response=raw,
    )
