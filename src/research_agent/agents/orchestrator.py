"""Orchestrator — routes commands and aggregates Analyst + Critic outputs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.analyst import AnalysisResult, Analyst, IdeaSupportResult
from research_agent.agents.base import AgentResponse
from research_agent.agents.critic import Critic, CritiqueResult
from research_agent.agents.debate import DebateHistory, DebateResult, FollowUpResult
from research_agent.agents.scribe import Draft, Scribe
from research_agent.agents.writing_pipeline import ReviewedDraft, WritingReview
from research_agent.core.language import DEFAULT_LANGUAGE
from research_agent.core.llm import LLMProvider
from research_agent.core.paper import Paper, Section
from research_agent.memory.working_memory import DEFAULT_MAX_CONTEXT_TOKENS, WorkingMemory
from research_agent.style.fingerprint import Fingerprint


@dataclass
class Task:
    """A routed unit of work for one or more agents."""

    command: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedAnalysis:
    """Dual-perspective paper analysis with consensus and conflict highlights."""

    analyst: AnalysisResult
    critic: CritiqueResult
    consensus: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    summary: str = ""

    def format_report(self) -> str:
        parts = [
            "# 📄 Analyst",
            self.analyst.to_agent_response().content,
            "",
            "# 🔴 Critic",
            self.critic.to_agent_response().content,
            "",
            "# 📊 Synthesis",
            self.summary,
        ]
        if self.consensus:
            parts.append("\n## Consensus")
            parts.extend(f"- {c}" for c in self.consensus)
        if self.conflicts:
            parts.append("\n## Conflicts / tensions")
            parts.extend(f"- {x}" for x in self.conflicts)
        return "\n".join(parts)


class Orchestrator:
    """Routes user commands and coordinates multi-agent workflows."""

    def __init__(self, llm: LLMProvider, *, language: str = DEFAULT_LANGUAGE) -> None:
        self.llm = llm
        self.language = language
        self.analyst = Analyst(llm, language=language)
        self.critic = Critic(llm, language=language)
        self.idea_analyst = Analyst(llm, language=language, prompt_stem="analyst_idea")
        self.idea_critic = Critic(llm, language=language, prompt_stem="critic_idea")
        self.idea_analyst_followup = Analyst(
            llm, language=language, prompt_stem="analyst_idea_followup"
        )
        self.idea_critic_followup = Critic(
            llm, language=language, prompt_stem="critic_idea_followup"
        )
        # M4 S4.3.1: dedicated writing-review system prompts.
        self.writing_analyst = Analyst(
            llm, language=language, prompt_stem="analyst_writing"
        )
        self.writing_critic = Critic(
            llm, language=language, prompt_stem="critic_writing"
        )

    def route(self, command: str, context: dict[str, Any] | None = None) -> Task:
        """Map a command string to a :class:`Task`."""
        ctx = dict(context or {})
        normalized = command.strip().lower().replace("-", "_")
        if normalized in {"read_paper", "read", "analyze_paper", "analyze"}:
            return Task(command="read_paper", context=ctx)
        if normalized in {"discuss", "discuss_idea"}:
            return Task(command="discuss_idea", context=ctx)
        return Task(command=normalized, context=ctx)

    async def run_task_async(self, task: Task) -> AggregatedAnalysis | dict[str, Any]:
        if task.command == "read_paper":
            return await self.analyze_paper_parallel(task.context["paper"])
        if task.command == "discuss_idea":
            memory = task.context.get("memory")
            if not isinstance(memory, WorkingMemory):
                raise TypeError("context['memory'] must be a WorkingMemory instance")
            max_tokens = int(task.context.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS))
            a, c = await self.discuss_turn_async(memory, max_context_tokens=max_tokens)
            return {"analyst": a, "critic": c}
        return {"command": task.command, "status": "not_implemented"}

    def run_task(self, task: Task) -> AggregatedAnalysis | dict[str, Any]:
        return asyncio.run(self.run_task_async(task))

    async def analyze_paper_parallel(self, paper: Paper) -> AggregatedAnalysis:
        """Run Analyst and Critic concurrently (``asyncio.gather``)."""
        if not isinstance(paper, Paper):
            raise TypeError("paper must be a Paper instance")

        analyst_result, critic_result = await asyncio.gather(
            asyncio.to_thread(self.analyst.analyze_paper, paper),
            asyncio.to_thread(self.critic.critique_paper, paper),
        )
        consensus, conflicts = _derive_consensus_conflicts(analyst_result, critic_result)
        summary = _build_summary(analyst_result, critic_result, consensus, conflicts)
        return AggregatedAnalysis(
            analyst=analyst_result,
            critic=critic_result,
            consensus=consensus,
            conflicts=conflicts,
            summary=summary,
        )

    async def discuss_turn_async(
        self,
        memory: WorkingMemory,
        *,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> tuple[AgentResponse, AgentResponse]:
        """Run Analyst + Critic using truncated session context from working memory."""
        paper = _paper_from_memory(memory, max_context_tokens=max_context_tokens)
        analyst_resp, critic_resp = await asyncio.gather(
            asyncio.to_thread(self.analyst.run, {"paper": paper}),
            asyncio.to_thread(self.critic.run, {"paper": paper}),
        )
        return analyst_resp, critic_resp

    async def debate_round_async(
        self,
        idea_text: str,
        context: str = "",
        *,
        paper: Paper | None = None,
        history: DebateHistory | None = None,
        target: str | None = None,
    ) -> DebateResult:
        """Run Idea debate (Analyst + Critic) or a targeted follow-up to one agent."""
        round_index = len(history.rounds) if history else 0
        prev = history.rounds[-1].result if history and history.rounds else None

        if target == "analyst":
            analyst = await asyncio.to_thread(
                self.idea_analyst.analyze_idea, idea_text, context, paper=paper
            )
            if prev is not None:
                return DebateResult.from_agent_results(
                    analyst,
                    _critique_from_debate(prev),
                    round_index=round_index,
                    score_delta=None,
                )
            critic = CritiqueResult(objections=[], support_score=5.0, score_reason="")
            return DebateResult.from_agent_results(analyst, critic, round_index=round_index)

        if target == "critic":
            critic = await asyncio.to_thread(
                self.idea_critic.critique_idea, idea_text, context, paper=paper
            )
            delta = history.score_delta(critic.support_score) if history else None
            if prev is not None:
                analyst = _support_from_debate(prev)
                return DebateResult.from_agent_results(
                    analyst,
                    critic,
                    round_index=round_index,
                    score_delta=delta,
                )
            analyst = IdeaSupportResult(supports=[], suggestions=[], evidence=[])
            return DebateResult.from_agent_results(
                analyst, critic, round_index=round_index, score_delta=delta
            )

        analyst_result, critic_result = await asyncio.gather(
            asyncio.to_thread(
                self.idea_analyst.analyze_idea, idea_text, context, paper=paper
            ),
            asyncio.to_thread(
                self.idea_critic.critique_idea, idea_text, context, paper=paper
            ),
        )
        delta = history.score_delta(critic_result.support_score) if history else None
        return DebateResult.from_agent_results(
            analyst_result,
            critic_result,
            round_index=round_index,
            score_delta=delta,
        )

    async def followup_turn_async(
        self,
        idea_text: str,
        user_message: str,
        context: str = "",
        *,
        paper: Paper | None = None,
        target: str | None = None,
    ) -> FollowUpResult:
        """Answer a follow-up in prose; requires a prior structured opening round."""
        if target == "analyst":
            analyst_text = await asyncio.to_thread(
                self.idea_analyst_followup.followup_idea,
                idea_text,
                user_message,
                context,
                paper=paper,
            )
            return FollowUpResult(
                analyst_conclusion=analyst_text,
                critic_conclusion="",
                targeted_agent="analyst",
            )
        if target == "critic":
            critic_text = await asyncio.to_thread(
                self.idea_critic_followup.followup_idea,
                idea_text,
                user_message,
                context,
                paper=paper,
            )
            return FollowUpResult(
                analyst_conclusion="",
                critic_conclusion=critic_text,
                targeted_agent="critic",
            )

        analyst_text, critic_text = await asyncio.gather(
            asyncio.to_thread(
                self.idea_analyst_followup.followup_idea,
                idea_text,
                user_message,
                context,
                paper=paper,
            ),
            asyncio.to_thread(
                self.idea_critic_followup.followup_idea,
                idea_text,
                user_message,
                context,
                paper=paper,
            ),
        )
        return FollowUpResult(
            analyst_conclusion=analyst_text,
            critic_conclusion=critic_text,
        )

    async def collect_writing_reviews_async(
        self,
        draft: Draft,
    ) -> list[WritingReview]:
        """Run analyst + critic writing reviews in parallel.

        Split out from :meth:`writing_review_pipeline_async` so the
        S4.3.2 interactive flow can present the reviews to the user
        and feed only the *selected* items into Scribe.revise.
        """
        analyst_review, critic_review = await asyncio.gather(
            asyncio.to_thread(
                self.writing_analyst.review_writing, draft.text, draft.section
            ),
            asyncio.to_thread(
                self.writing_critic.review_writing, draft.text, draft.section
            ),
        )
        return [analyst_review, critic_review]

    def collect_writing_reviews(self, draft: Draft) -> list[WritingReview]:
        """Sync wrapper around :meth:`collect_writing_reviews_async`."""
        return asyncio.run(self.collect_writing_reviews_async(draft))

    async def writing_review_pipeline_async(
        self,
        scribe: Scribe,
        draft: Draft,
        *,
        fingerprint: Fingerprint | None = None,
    ) -> ReviewedDraft:
        """Run the full Scribe → Analyst+Critic → Scribe cycle.

        Reviews run in parallel via ``asyncio.gather``. The revision
        step is a single LLM call after both reviews finish.
        """
        reviews = await self.collect_writing_reviews_async(draft)
        revised = await asyncio.to_thread(
            scribe.revise, draft, reviews, fingerprint=fingerprint
        )
        return ReviewedDraft(original=draft, reviews=reviews, revised=revised)

    def writing_review_pipeline(
        self,
        scribe: Scribe,
        draft: Draft,
        *,
        fingerprint: Fingerprint | None = None,
    ) -> ReviewedDraft:
        """Sync wrapper around :meth:`writing_review_pipeline_async`."""
        return asyncio.run(
            self.writing_review_pipeline_async(scribe, draft, fingerprint=fingerprint)
        )

    def extract_search_context(
        self,
        memory: WorkingMemory,
        *,
        max_messages: int = 12,
        max_chars: int = 3000,
    ) -> str:
        """Build a compact transcript snippet for ``Searcher.suggest_refinement``.

        Keeps the most recent ``max_messages`` messages (any role) and
        formats them ``role: content``, capped at ``max_chars``. The
        Searcher prompt is responsible for *interpreting* this into a
        refined query; the orchestrator just gives it well-shaped raw
        material.

        Returns an empty string when there's nothing useful (no messages
        or all-empty content).
        """
        if not memory.messages:
            return ""
        recent = memory.messages[-max_messages:]
        lines: list[str] = []
        for msg in recent:
            content = (msg.content or "").strip()
            if not content:
                continue
            # Tighten very long messages; Searcher only needs gist.
            if len(content) > 600:
                content = content[:600].rstrip() + "…"
            lines.append(f"{msg.role}: {content}")
        if not lines:
            return ""
        text = "\n".join(lines)
        if len(text) > max_chars:
            # Keep the tail (most recent), not the head.
            text = "…\n" + text[-max_chars:]
        return text


def _support_from_debate(prev: DebateResult) -> IdeaSupportResult:
    return IdeaSupportResult(
        supports=list(prev.supports),
        suggestions=list(prev.suggestions),
        evidence=[],
        confidence=prev.confidence,
    )


def _critique_from_debate(prev: DebateResult) -> CritiqueResult:
    return CritiqueResult(
        objections=list(prev.objections),
        support_score=prev.score,
        score_reason=prev.score_reason,
        suggestions=list(prev.suggestions),
    )


def _paper_from_memory(memory: WorkingMemory, *, max_context_tokens: int) -> Paper:
    """Build a synthetic Paper so discuss can reuse analyze/critique prompts."""
    transcript = memory.to_context(max_context_tokens)
    last_user = ""
    for msg in reversed(memory.messages):
        if msg.role == "user":
            last_user = msg.content
            break
    return Paper(
        id=f"session:{memory.session_id}",
        title="Research discussion",
        abstract=last_user,
        sections=[Section(title="Conversation", content=transcript)],
        full_text=transcript,
    )


def _derive_consensus_conflicts(
    analyst: AnalysisResult,
    critic: CritiqueResult,
) -> tuple[list[str], list[str]]:
    consensus: list[str] = []
    conflicts: list[str] = []

    if analyst.contributions and critic.support_score >= 7:
        consensus.append("Analyst sees meaningful contributions; Critic rates support ≥ 7.")
    if analyst.confidence >= 0.6 and critic.support_score >= 8:
        consensus.append("Both perspectives indicate a relatively strong paper.")

    if critic.support_score < 7 and analyst.confidence >= 0.7:
        conflicts.append("Analyst is fairly confident while Critic support score is below 7.")
    if critic.objections and analyst.contributions:
        top_objection = critic.objections[0][:120]
        top_contribution = analyst.contributions[0][:120]
        conflicts.append(
            f"Critic challenges ('{top_objection}…') vs Analyst highlight "
            f"('{top_contribution}…')."
        )
    if critic.support_score <= 5 and analyst.potential_impact:
        conflicts.append("Critic is skeptical despite Analyst noting non-trivial potential impact.")

    return consensus, conflicts


def _build_summary(
    analyst: AnalysisResult,
    critic: CritiqueResult,
    consensus: list[str],
    conflicts: list[str],
) -> str:
    n_contrib = len(analyst.contributions)
    n_obj = len(critic.objections)
    tone = "balanced"
    if critic.support_score >= 8 and analyst.confidence >= 0.6:
        tone = "mostly positive"
    elif critic.support_score < 6:
        tone = "skeptical"

    lines = [
        f"Dual review of the paper: {n_contrib} contribution(s) highlighted, "
        f"{n_obj} objection(s) raised. Critic support {critic.support_score:.0f}/9. "
        f"Overall tone: {tone}.",
    ]
    if consensus:
        lines.append(f"Alignment: {consensus[0]}")
    if conflicts:
        lines.append(f"Key tension: {conflicts[0]}")
    return " ".join(lines)
