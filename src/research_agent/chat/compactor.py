"""Rolling history compaction (T2.1).

When a session grows past ``COMPACT_TRIGGER_THRESHOLD`` live messages,
fold the oldest ``len - COMPACT_KEEP_RECENT`` into a single dense
``session_summary`` entry so long-range context survives without paying
for the full transcript on every turn. Tagged messages stay in
``WorkingMemory`` for ``/history`` — only the LLM prompt drops them.

Trigger is lazy (call at turn start). One extra LLM call ≈ once every
``COMPACT_TRIGGER_THRESHOLD - COMPACT_KEEP_RECENT`` user turns; the
cost is dwarfed by the savings on subsequent turns that would
otherwise replay 30+ messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.core.llm import ChatMessage, LLMError, LLMProvider
from research_agent.memory.working_memory import MemoryMessage

if TYPE_CHECKING:
    from research_agent.chat.session import ChatSession

# Filter kinds that exclude messages from the live conversational history.
# Keep in sync with :func:`research_agent.chat.router._build_messages`.
EXCLUDED_KINDS = frozenset(
    {"tool_log", "compacted", "session_summary", "scratchpad"}
)

# Trigger once we have this many live messages.
COMPACT_TRIGGER_THRESHOLD = 16

# Always leave this many recent messages untouched after compaction.
COMPACT_KEEP_RECENT = 8

# Hard cap on the summary so it never balloons the context budget.
SUMMARY_MAX_CHARS = 1200

# Cap individual lines we feed to the summarizer; full content is already
# in WorkingMemory for /history, we just need enough signal for a recap.
TRANSCRIPT_LINE_MAX_CHARS = 600


_SUMMARY_SYSTEM_PROMPT = (
    "You compress a research-assistant session into a concise summary "
    "so a chat agent can pick up where the user left off. Keep it under "
    "150 words. Capture: (1) what paper(s)/topic the user is exploring, "
    "(2) key claims or critiques discussed, (3) decisions made (saved "
    "ideas, drafts written, queued papers), (4) open threads to revisit. "
    "Drop pleasantries, repeated tool-call traces, and verbatim quotes "
    "longer than one sentence. Reply with the summary only — no preamble."
)


def _live_messages(session: ChatSession) -> list[MemoryMessage]:
    """Messages that count toward live conversational history."""
    return [
        m for m in session.memory.messages
        if m.metadata.get("kind") not in EXCLUDED_KINDS
    ]


def _format_for_summarizer(messages: list[MemoryMessage]) -> str:
    """Render messages in a compact role-prefixed transcript."""
    lines: list[str] = []
    for m in messages:
        text = m.content.strip().replace("\n", " ")
        if len(text) > TRANSCRIPT_LINE_MAX_CHARS:
            text = text[:TRANSCRIPT_LINE_MAX_CHARS - 1] + "…"
        lines.append(f"{m.role}: {text}")
    return "\n".join(lines)


def _summarize(llm: LLMProvider, transcript: str) -> str:
    """One cheap non-streaming call. Bounded by ``SUMMARY_MAX_CHARS``."""
    messages = [
        ChatMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "Summarize the following session transcript:\n\n"
                + transcript
            ),
        ),
    ]
    try:
        text = llm.chat(messages, temperature=0.2, max_tokens=400)
    except LLMError:
        # Compaction must never break the live agent loop; on failure we
        # just skip and keep the originals.
        return ""
    text = text.strip()
    if len(text) > SUMMARY_MAX_CHARS:
        text = text[:SUMMARY_MAX_CHARS - 1] + "…"
    return text


def maybe_compact_history(
    session: ChatSession,
    llm: LLMProvider,
    *,
    trigger_threshold: int = COMPACT_TRIGGER_THRESHOLD,
    keep_recent: int = COMPACT_KEEP_RECENT,
) -> bool:
    """Compact oldest live history when over threshold.

    Returns ``True`` when a summary was written, ``False`` otherwise.
    Idempotent — re-running after a successful compaction is a no-op
    until ``trigger_threshold`` more live messages accumulate.
    """
    live = _live_messages(session)
    if len(live) <= trigger_threshold:
        return False

    to_summarize = live[:-keep_recent] if keep_recent > 0 else list(live)
    if not to_summarize:
        return False

    transcript = _format_for_summarizer(to_summarize)
    summary = _summarize(llm, transcript)
    if not summary:
        return False

    # Persist the summary as a regular memory entry. _build_messages
    # hoists ``session_summary`` entries to the top of the prompt.
    session.memory.append(
        "system",
        f"[earlier session summary] {summary}",
        metadata={"kind": "session_summary"},
    )
    # Tag originals so the next ``_build_messages`` skips them. They
    # remain queryable via ``/history`` and persist to discussions.db
    # on session close.
    for m in to_summarize:
        m.metadata["kind"] = "compacted"
    return True
