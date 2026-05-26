"""Memory Keeper — recalls related historical ideas + discussions."""

from __future__ import annotations

from research_agent.core.idea import Idea
from research_agent.storage.discussion_vectors import (
    INDEXABLE_ROLES,
    DiscussionVectorStore,
)
from research_agent.storage.discussions import DiscussionMessage, DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore

LOW_SCORE_THRESHOLD = 5.0


class MemoryKeeper:
    """Surface similar past ideas, discussions, and low-score warnings."""

    def __init__(
        self,
        ideas: IdeaRepository,
        vectors: IdeaVectorStore,
        *,
        discussions: DiscussionRepository | None = None,
        discussion_vectors: DiscussionVectorStore | None = None,
    ) -> None:
        self.ideas = ideas
        self.vectors = vectors
        self.discussions = discussions
        self.discussion_vectors = discussion_vectors

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

    # ------------------------- discussion recall -------------------------

    def index_message(
        self,
        *,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        idea_id: str | None = None,
    ) -> None:
        """Index a single persisted DiscussionMessage if its role is relevant."""
        if self.discussion_vectors is None or role not in INDEXABLE_ROLES:
            return
        self.discussion_vectors.upsert(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            idea_id=idea_id,
        )

    def recall_history(
        self,
        query: str,
        *,
        limit: int = 5,
        exclude_session_id: str | None = None,
    ) -> list[DiscussionMessage]:
        """Return past DiscussionMessages most similar to ``query``.

        Returns ``[]`` if discussion vectors/repo are unavailable. The
        ``exclude_session_id`` knob lets callers skip the current REPL session
        so recall is genuinely cross-session.
        """
        if self.discussion_vectors is None or self.discussions is None:
            return []
        ids = self.discussion_vectors.query_similar(
            query, limit=limit, exclude_session_id=exclude_session_id
        )
        if not ids:
            return []
        by_id = self.discussions.get_many(ids)
        # Preserve similarity order.
        return [by_id[mid] for mid in ids if mid in by_id]
