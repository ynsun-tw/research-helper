"""Memory Keeper — recalls related historical ideas + discussions."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.core.idea import Idea, IdeaStatus
from research_agent.storage.discussion_vectors import (
    INDEXABLE_ROLES,
    DiscussionVectorStore,
)
from research_agent.storage.discussions import DiscussionMessage, DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore

LOW_SCORE_THRESHOLD = 5.0

# Statuses worth re-surfacing as proactive alerts. "active" is excluded -
# the user is already aware of those; "shelved" and "waiting" are the
# ideas they parked and might want to reconsider when a related paper
# walks back into the conversation. M3 spec S3.4.1 (acceptance criterion
# 1) calls these out by name.
DEFAULT_ASSOCIATION_STATUSES: tuple[IdeaStatus, ...] = ("shelved", "waiting")
DEFAULT_ASSOCIATION_THRESHOLD = 0.8


@dataclass(frozen=True)
class Association:
    """A historical idea surfaced by check_associations.

    ``similarity`` is in ``[0.0, 1.0]`` (higher means more related);
    callers can use it to decide whether to surface or suppress the
    alert without re-querying the vector store.
    """

    idea: Idea
    similarity: float


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

    # ----------------------- proactive associations (T3.4.1.1) -----------

    def check_associations(
        self,
        context: str,
        *,
        threshold: float = DEFAULT_ASSOCIATION_THRESHOLD,
        limit: int = 5,
        statuses: tuple[IdeaStatus, ...] = DEFAULT_ASSOCIATION_STATUSES,
    ) -> list[Association]:
        """Return parked/shelved ideas semantically close to ``context``.

        Designed as a low-friction "is this paper we're looking at related
        to something we set aside?" probe the Orchestrator can call
        cheaply after /read or /discuss. Behaviour:

        - Vector-store the ``context`` and pull the top-K candidates,
          over-fetching by 3x so the post-filter (status, threshold)
          still leaves enough room to fill ``limit``.
        - Drop any candidate whose similarity is < ``threshold``
          ([0.0, 1.0]; M3 spec target is 0.8 - configurable per call
          and via ``memory.alert_threshold`` in :class:`Config`).
        - Drop candidates whose status isn't in ``statuses`` (default
          ``("shelved", "waiting")``). Pass ``statuses=()`` to bypass
          the status filter entirely.
        - Skip orphaned vector rows (the SQLite row was deleted but the
          vector hadn't been removed yet).

        Returns at most ``limit`` Associations in similarity order.
        Empty string / empty store returns ``[]``.
        """
        if not context.strip():
            return []
        overfetch = max(limit * 3, limit)
        pairs = self.vectors.query_with_scores(context, limit=overfetch)
        out: list[Association] = []
        for iid, score in pairs:
            if score < threshold:
                continue
            idea = self.ideas.get(iid)
            if idea is None:
                continue
            if statuses and idea.status not in statuses:
                continue
            out.append(Association(idea=idea, similarity=score))
            if len(out) >= limit:
                break
        return out

    def format_associations(self, associations: list[Association]) -> str:
        """Compact Rich-markup banner for the Orchestrator to inject.

        Empty input -> empty string so the caller can ``if banner:``
        check before printing. Format is deliberately quiet (one
        leading line + one bullet per association) so the alert
        doesn't dominate the user's screen during a normal /read.
        """
        if not associations:
            return ""
        lines = [
            "[bold yellow]Related ideas you parked previously[/bold yellow]"
        ]
        for assoc in associations:
            idea = assoc.idea
            lines.append(
                f"- [cyan]{idea.title}[/cyan] "
                f"({idea.status}, similarity {assoc.similarity:.0%}) — "
                f"[dim]/idea show {idea.id[:8]}[/dim]"
            )
        return "\n".join(lines)

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
