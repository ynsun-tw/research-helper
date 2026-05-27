"""Structured debate results and multi-round history."""

from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.agents.analyst import IdeaSupportResult
from research_agent.agents.critic import CritiqueResult


@dataclass
class DebateResult:
    """Fixed-structure output for Idea debate rounds."""

    supports: list[str]
    objections: list[str]
    suggestions: list[str]
    score: float
    score_reason: str
    confidence: float
    round_index: int = 0
    score_delta: float | None = None

    def to_markdown(self) -> str:
        lines = [
            f"## Support score: {self.score:.0f}/9",
            f"**Reason:** {self.score_reason or '(not provided)'}",
            "",
            "## Supports",
            *(_bullet_lines(self.supports) or ["- (none)"]),
            "",
            "## Objections",
            *(_bullet_lines(self.objections) or ["- (none)"]),
            "",
            "## Suggestions",
            *(_bullet_lines(self.suggestions) or ["- (none)"]),
            "",
            f"**Analyst confidence:** {self.confidence:.0%}",
        ]
        if self.score_delta is not None:
            sign = "+" if self.score_delta >= 0 else ""
            lines.append(f"\n**Score change:** {sign}{self.score_delta:.0f} vs previous round")
        return "\n".join(lines)

    @classmethod
    def from_agent_results(
        cls,
        analyst: IdeaSupportResult,
        critic: CritiqueResult,
        *,
        round_index: int = 0,
        score_delta: float | None = None,
    ) -> DebateResult:
        suggestions = list(analyst.suggestions)
        for s in critic.suggestions:
            if s not in suggestions:
                suggestions.append(s)
        return cls(
            supports=analyst.supports,
            objections=critic.objections,
            suggestions=suggestions,
            score=critic.support_score,
            score_reason=critic.score_reason,
            confidence=analyst.confidence,
            round_index=round_index,
            score_delta=score_delta,
        )


@dataclass
class FollowUpResult:
    """Prose conclusions for follow-up turns (no structured debate schema)."""

    analyst_conclusion: str
    critic_conclusion: str
    targeted_agent: str | None = None


@dataclass
class DebateRound:
    user_message: str
    result: DebateResult | None = None
    followup: FollowUpResult | None = None
    targeted_agent: str | None = None


@dataclass
class DebateHistory:
    """In-session debate state across multiple rounds."""

    rounds: list[DebateRound] = field(default_factory=list)
    idea_id: str | None = None

    def append(
        self,
        user_message: str,
        result: DebateResult,
        *,
        targeted_agent: str | None = None,
    ) -> None:
        self.rounds.append(
            DebateRound(
                user_message=user_message,
                result=result,
                targeted_agent=targeted_agent,
            )
        )

    def append_followup(
        self,
        user_message: str,
        followup: FollowUpResult,
        *,
        targeted_agent: str | None = None,
    ) -> None:
        self.rounds.append(
            DebateRound(
                user_message=user_message,
                followup=followup,
                targeted_agent=targeted_agent,
            )
        )

    @property
    def has_initial_round(self) -> bool:
        return any(r.result is not None for r in self.rounds)

    def last_score(self) -> float | None:
        for rnd in reversed(self.rounds):
            if rnd.result is not None:
                return rnd.result.score
        return None

    def score_delta(self, new_score: float) -> float | None:
        prev = self.last_score()
        if prev is None:
            return None
        return new_score - prev


def _bullet_lines(items: list[str]) -> list[str]:
    if not items:
        return []
    return [f"- {item}" for item in items]
