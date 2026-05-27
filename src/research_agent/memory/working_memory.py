"""In-session working memory for multi-turn agent conversations."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from research_agent.storage.discussions import DiscussionRepository

# Rough English heuristic (~4 chars per token); avoids optional tiktoken dependency.
CHARS_PER_TOKEN = 4
DEFAULT_MAX_CONTEXT_TOKENS = 8000


def estimate_tokens(text: str) -> int:
    """Approximate token count for context budgeting."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class MemoryMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemory:
    """Holds the current session transcript; truncates for LLM context."""

    session_id: str
    messages: list[MemoryMessage] = field(default_factory=list)
    idea_id: str | None = None
    _persisted_count: int = field(default=0, repr=False)

    @classmethod
    def new_session(cls, session_id: str | None = None) -> WorkingMemory:
        return cls(session_id=session_id or str(uuid.uuid4()))

    @classmethod
    def from_session(cls, repo: DiscussionRepository, session_id: str) -> WorkingMemory:
        """Restore in-memory state from persisted ``discussions`` rows."""
        memory = cls(session_id=session_id)
        for row in repo.list_session(session_id):
            memory.messages.append(
                MemoryMessage(role=row.role, content=row.content, metadata=row.metadata)
            )
        memory._persisted_count = len(memory.messages)
        return memory

    def append(
        self,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.messages.append(
            MemoryMessage(role=role, content=content, metadata=dict(metadata or {}))
        )

    def to_context(self, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> str:
        """Format recent messages for the LLM, dropping oldest lines when over budget."""
        if not self.messages:
            return ""
        if max_tokens <= 0:
            return ""

        lines: list[str] = []
        used = 0
        for msg in reversed(self.messages):
            line = f"{msg.role}: {msg.content}"
            line_tokens = estimate_tokens(line) + 1  # newline
            if used + line_tokens > max_tokens and lines:
                break
            lines.append(line)
            used += line_tokens

        lines.reverse()
        return "\n".join(lines)

    def persist(
        self,
        repo: DiscussionRepository,
        *,
        indexer: Callable[[str, MemoryMessage], None] | None = None,
    ) -> int:
        """Write any in-memory messages not yet stored in SQLite.

        ``indexer`` (if given) is called with ``(message_id, message)`` for
        every newly persisted row, letting callers (e.g. MemoryKeeper) push
        the same content into a vector store in lock-step with SQLite.
        """
        written = 0
        for msg in self.messages[self._persisted_count :]:
            mid = repo.append(
                self.session_id,
                msg.role,
                msg.content,
                metadata=msg.metadata or None,
                idea_id=self.idea_id,
            )
            if indexer is not None:
                indexer(mid, msg)
            written += 1
        self._persisted_count = len(self.messages)
        return written

    def turn_count(self) -> int:
        """Number of user turns in this session."""
        return sum(1 for m in self.messages if m.role == "user")
