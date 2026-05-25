"""Orchestrator — routes commands and aggregates Analyst + Critic outputs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.analyst import AnalysisResult, Analyst
from research_agent.agents.base import AgentResponse
from research_agent.agents.critic import Critic, CritiqueResult
from research_agent.core.llm import LLMProvider
from research_agent.core.paper import Paper, Section


@dataclass(slots=True)
class Task:
    """A routed unit of work for one or more agents."""

    command: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
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

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.analyst = Analyst(llm)
        self.critic = Critic(llm)

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
            message = str(task.context.get("message", ""))
            history = task.context.get("history") or []
            a, c = await self.discuss_turn_async(message, history)
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
        message: str,
        history: list[tuple[str, str]],
    ) -> tuple[AgentResponse, AgentResponse]:
        """Run Analyst + Critic on a discuss turn (reuses paper analysis pipeline)."""
        paper = _paper_from_discussion(message, history)
        analyst_resp, critic_resp = await asyncio.gather(
            asyncio.to_thread(self.analyst.run, {"paper": paper}),
            asyncio.to_thread(self.critic.run, {"paper": paper}),
        )
        return analyst_resp, critic_resp


def _paper_from_discussion(message: str, history: list[tuple[str, str]]) -> Paper:
    """Build a synthetic Paper so discuss can reuse analyze/critique prompts."""
    transcript_lines = [f"{role}: {content}" for role, content in history[-12:]]
    transcript_lines.append(f"user: {message}")
    transcript = "\n".join(transcript_lines)
    return Paper(
        id="session:discuss",
        title="Research discussion",
        abstract=message,
        sections=[
            Section(
                title="Conversation",
                content=transcript,
            )
        ],
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
            f"Critic challenges ('{top_objection}…') vs Analyst highlight ('{top_contribution}…')."
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
