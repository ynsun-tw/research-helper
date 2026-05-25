"""Memory Keeper — recalls related historical ideas at session start."""

from __future__ import annotations

from research_agent.core.idea import Idea
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore

LOW_SCORE_THRESHOLD = 5.0


class MemoryKeeper:
    """Surface similar past ideas and low-score warnings before a debate."""

    def __init__(
        self,
        ideas: IdeaRepository,
        vectors: IdeaVectorStore,
    ) -> None:
        self.ideas = ideas
        self.vectors = vectors

    def recall_similar(
        self,
        idea_text: str,
        *,
        limit: int = 3,
        exclude_id: str | None = None,
    ) -> list[Idea]:
        ids = self.vectors.query_similar(idea_text, limit=limit, exclude_id=exclude_id)
        found: list[Idea] = []
        for iid in ids:
            idea = self.ideas.get(iid)
            if idea is not None:
                found.append(idea)
        return found

    def format_reminders(self, similar: list[Idea], *, query_text: str = "") -> str:
        if not similar:
            return ""
        lines = ["[bold yellow]Related ideas from your library[/bold yellow]"]
        for idea in similar:
            score = idea.critic_score
            score_txt = f"{score:.0f}/9" if score is not None else "—"
            lines.append(f"- [cyan]{idea.title}[/cyan] ({idea.status}, last score {score_txt})")
            if idea.description:
                snippet = idea.description[:120].replace("\n", " ")
                lines.append(f"  {snippet}…")
        low = [
            i
            for i in similar
            if i.critic_score is not None and i.critic_score < LOW_SCORE_THRESHOLD
        ]
        if low:
            titles = ", ".join(i.title for i in low)
            lines.append(
                f"\n[bold red]Note:[/bold red] Similar idea(s) scored below "
                f"{LOW_SCORE_THRESHOLD:.0f}: {titles}. Consider addressing past objections."
            )
        return "\n".join(lines)

    def index_idea(self, idea: Idea) -> None:
        self.vectors.upsert(idea)
