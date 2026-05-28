"""In-session working memory for multi-turn agent conversations."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from research_agent.storage.discussions import DiscussionRepository

# Rough English heuristic (~4 chars per token); used as fallback when
# tiktoken is not installed.
CHARS_PER_TOKEN = 4
DEFAULT_MAX_CONTEXT_TOKENS = 8000

# T3.1: opportunistic tiktoken integration. We use a single shared
# encoder (cl100k_base is OpenAI's default for GPT-3.5/4 and lines up
# closely with most providers we target). When tiktoken isn't
# installed — it's an optional dep — we transparently fall back to the
# CHARS_PER_TOKEN heuristic so production never depends on the binary.
try:  # pragma: no cover - import-side branch
    import tiktoken as _tiktoken  # type: ignore[import-not-found]

    _ENCODER: Any = _tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken absent or load failed
    _ENCODER = None

# Conservative known context-window sizes (input + output), keyed by a
# substring match on the model slug. The first match wins, so list the
# more specific patterns first. When nothing matches we fall back to
# ``DEFAULT_MAX_CONTEXT_TOKENS`` (8K) — a safe lower bound that won't
# blow up the prompt on an unknown small-model.
#
# Sources: vendor docs (Anthropic, OpenAI, DeepSeek, OpenRouter), 2026-05.
# These are *input* windows; the actual prompt budget is reduced further
# by ``reserve_tokens_for_output`` so we always leave room for the reply.
MODEL_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-7", 200_000),
    ("claude-4", 200_000),
    ("claude", 200_000),
    ("gpt-4o", 128_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("gemini-1.5", 1_000_000),
    ("gemini-2", 1_000_000),
    ("gemini", 32_000),
    ("deepseek-chat", 64_000),
    ("deepseek-reasoner", 64_000),
    ("deepseek-coder", 16_000),
    ("deepseek", 64_000),
    ("mistral-large", 128_000),
    ("mixtral", 32_000),
    ("llama-3", 128_000),
    ("qwen", 32_000),
    ("kimi", 200_000),
)

DEFAULT_RESERVE_OUTPUT_TOKENS = 4000


def estimate_tokens(text: str) -> int:
    """Approximate token count for context budgeting.

    Uses ``tiktoken`` (cl100k_base) when available — it's a few percent
    off for non-OpenAI providers but vastly better than the 4-char
    heuristic, especially for short messages, code, and CJK text.
    Falls back to ``len(text) // CHARS_PER_TOKEN`` when tiktoken isn't
    installed.
    """
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return max(1, len(_ENCODER.encode(text)))
        except Exception:  # pragma: no cover - defensive
            pass
    return max(1, len(text) // CHARS_PER_TOKEN)


def lookup_model_context_window(model: str) -> int:
    """Best-effort context window for a model slug.

    Returns ``DEFAULT_MAX_CONTEXT_TOKENS`` (conservative) on no match
    so an unknown small model never gets a giant prompt that overflows.
    Matching is case-insensitive substring; the table is ordered most
    specific → least specific.
    """
    if not model:
        return DEFAULT_MAX_CONTEXT_TOKENS
    needle = model.lower()
    for key, window in MODEL_CONTEXT_WINDOWS:
        if key in needle:
            return window
    return DEFAULT_MAX_CONTEXT_TOKENS


def resolve_context_budget(
    model: str,
    *,
    override: int = 0,
    reserve_output: int = DEFAULT_RESERVE_OUTPUT_TOKENS,
) -> int:
    """Effective input-token budget for a session.

    - ``override > 0`` pins the cap (useful for tests / cost control).
    - Otherwise we look up the model's known window and subtract
      ``reserve_output`` so the reply has space to land.
    - The result is bounded below by ``DEFAULT_MAX_CONTEXT_TOKENS`` to
      stay backwards-compatible with the old fixed 8K behaviour: a
      model with a tiny window won't shrink the budget below what the
      existing logic assumed.
    """
    raw = override if override > 0 else lookup_model_context_window(model)
    effective = raw - max(0, reserve_output)
    return max(effective, DEFAULT_MAX_CONTEXT_TOKENS)


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
