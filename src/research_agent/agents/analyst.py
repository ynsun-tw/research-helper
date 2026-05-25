"""Analyst agent — curiosity-driven paper understanding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent, extract_json
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
    data = extract_json(raw)
    pairs = []
    for item in data.get("claimed_vs_evidence") or []:
        if isinstance(item, dict):
            pairs.append(
                ClaimedVsEvidence(
                    claim=str(item.get("claim", "")),
                    evidence=str(item.get("evidence", "")),
                )
            )
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    return AnalysisResult(
        contributions=_as_str_list(data.get("contributions")),
        method_insights=_as_str_list(data.get("method_insights")),
        potential_impact=str(data.get("potential_impact", "")),
        related_work=_as_str_list(data.get("related_work")),
        claimed_vs_evidence=pairs,
        confidence=confidence,
        raw_response=raw,
    )


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v]
