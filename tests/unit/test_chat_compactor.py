"""Tests for rolling history compaction (T2.1)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from research_agent.chat.compactor import (
    COMPACT_KEEP_RECENT,
    COMPACT_TRIGGER_THRESHOLD,
    maybe_compact_history,
)
from research_agent.chat.router import _build_messages
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


def _cfg(d: Path) -> Config:
    return Config(data_dir=d, api_key="sk-x")


def _make_session(d: Path, llm: MockLLMProvider) -> ChatSession:
    return ChatSession.create(
        cfg=_cfg(d),
        llm=llm,
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )


def test_no_compaction_below_threshold(config_dir: Path) -> None:
    llm = MockLLMProvider(["should not be called"])
    session = _make_session(config_dir, llm)
    # Stay strictly below the threshold (count pairs as 2 each).
    pairs = (COMPACT_TRIGGER_THRESHOLD // 2) - 1
    for i in range(pairs):
        session.memory.append("user", f"turn {i}")
        session.memory.append("assistant", f"reply {i}")

    fired = maybe_compact_history(session, llm)
    assert fired is False
    # Nothing should have been tagged or summarized.
    assert all(
        m.metadata.get("kind") != "compacted" for m in session.memory.messages
    )


def test_compaction_summarizes_oldest_and_keeps_recent(
    config_dir: Path,
) -> None:
    llm = MockLLMProvider(["User explored the Transformer paper; saved one idea."])
    session = _make_session(config_dir, llm)
    # Trigger threshold + a couple more so we're sure we exceed it.
    total = COMPACT_TRIGGER_THRESHOLD + 4
    for i in range(total):
        session.memory.append("user", f"q{i}")
        session.memory.append("assistant", f"a{i}")

    fired = maybe_compact_history(session, llm)
    assert fired is True

    # Exactly one session_summary entry exists.
    summaries = [
        m for m in session.memory.messages
        if m.metadata.get("kind") == "session_summary"
    ]
    assert len(summaries) == 1
    assert "Transformer" in summaries[0].content

    # The most recent COMPACT_KEEP_RECENT live messages remain untagged.
    live = [
        m for m in session.memory.messages
        if m.metadata.get("kind") is None
    ]
    assert len(live) == COMPACT_KEEP_RECENT


def test_build_messages_hoists_summary_above_history(
    config_dir: Path,
) -> None:
    llm = MockLLMProvider(["summary: user is working on attention mechanisms."])
    session = _make_session(config_dir, llm)
    for i in range(COMPACT_TRIGGER_THRESHOLD + 4):
        session.memory.append("user", f"q{i}")
        session.memory.append("assistant", f"a{i}")
    maybe_compact_history(session, llm)

    msgs = _build_messages(session)
    contents = [m.content for m in msgs]
    summary_text = "earlier session summary"
    assert any(summary_text in c for c in contents)
    # Summary must appear BEFORE any live history message.
    summary_idx = next(
        i for i, c in enumerate(contents) if summary_text in c
    )
    recent_idx = next(
        i for i, c in enumerate(contents)
        if c.startswith("q") or c.startswith("a")
    )
    assert summary_idx < recent_idx

    # The compacted originals must NOT be in the prompt.
    for c in contents:
        # Early-turn placeholders like "q0", "a0" are excluded; we only
        # see them if compaction failed to filter them.
        assert c not in {"q0", "a0", "q1", "a1"}


def test_compaction_is_idempotent(config_dir: Path) -> None:
    llm = MockLLMProvider(["summary 1", "summary 2"])
    session = _make_session(config_dir, llm)
    for i in range(COMPACT_TRIGGER_THRESHOLD + 4):
        session.memory.append("user", f"q{i}")
        session.memory.append("assistant", f"a{i}")
    assert maybe_compact_history(session, llm) is True
    # Second call: live count is now COMPACT_KEEP_RECENT (<=threshold) → no-op.
    assert maybe_compact_history(session, llm) is False


def test_compaction_swallows_llm_failure(config_dir: Path) -> None:
    from research_agent.core.llm import LLMError

    class FailingLLM(MockLLMProvider):
        def chat(self, messages, *, temperature=0.7, max_tokens=None):  # type: ignore[no-untyped-def, override]
            raise LLMError("simulated outage")

    llm = FailingLLM([])
    session = _make_session(config_dir, llm)
    for i in range(COMPACT_TRIGGER_THRESHOLD + 4):
        session.memory.append("user", f"q{i}")
        session.memory.append("assistant", f"a{i}")
    # Must not raise; just returns False because no summary was written.
    assert maybe_compact_history(session, llm) is False
    # And no originals were tagged compacted (we don't want partial state).
    assert not any(
        m.metadata.get("kind") == "compacted" for m in session.memory.messages
    )


def test_compactor_filters_tool_log_from_count(config_dir: Path) -> None:
    """tool_log entries don't count toward the live-history threshold,
    because they're already excluded from prompts. Otherwise heavy
    /insights / /search usage would trigger compaction spuriously."""
    llm = MockLLMProvider(["should not be called"])
    session = _make_session(config_dir, llm)
    # Fill the session with tool_log messages well beyond the threshold.
    for i in range(COMPACT_TRIGGER_THRESHOLD * 3):
        session.memory.append(
            "system", f"tool noise {i}", metadata={"kind": "tool_log"}
        )
    # Add only a handful of real turns.
    session.memory.append("user", "hello")
    session.memory.append("assistant", "hi")

    assert maybe_compact_history(session, llm) is False


def test_compactor_skips_when_keep_recent_swallows_everything(
    config_dir: Path,
) -> None:
    """If the threshold is met but the ``keep_recent`` window already
    covers every live message (degenerate config), do nothing."""
    llm = MockLLMProvider(["x"])
    session = _make_session(config_dir, llm)
    for i in range(20):
        session.memory.append("user", f"q{i}")
    # With keep_recent >= live count, to_summarize is empty.
    assert (
        maybe_compact_history(
            session, llm, trigger_threshold=5, keep_recent=100
        )
        is False
    )


def test_summary_chat_call_uses_low_temperature(config_dir: Path) -> None:
    """The compactor should call the LLM with a low temperature (deterministic
    summarization) and a small max_tokens budget."""
    calls: list[tuple[float, int | None]] = []

    class RecordingLLM(MockLLMProvider):
        def chat(  # type: ignore[override]
            self,
            messages,
            *,
            temperature: float = 0.7,
            max_tokens: int | None = None,
        ) -> str:
            calls.append((temperature, max_tokens))
            return "ok"

    llm = RecordingLLM([])
    session = _make_session(config_dir, llm)
    for i in range(COMPACT_TRIGGER_THRESHOLD + 4):
        session.memory.append("user", f"q{i}")
    maybe_compact_history(session, llm)
    assert calls, "compactor never called the LLM"
    temp, mt = calls[-1]
    assert temp <= 0.3
    assert mt is not None and mt <= 600


def test_compactor_caps_oversized_summary(config_dir: Path) -> None:
    from research_agent.chat.compactor import SUMMARY_MAX_CHARS

    big = "Z" * (SUMMARY_MAX_CHARS + 5000)
    llm = MockLLMProvider([big])
    session = _make_session(config_dir, llm)
    for i in range(COMPACT_TRIGGER_THRESHOLD + 4):
        session.memory.append("user", f"q{i}")
    maybe_compact_history(session, llm)
    summary = next(
        m for m in session.memory.messages
        if m.metadata.get("kind") == "session_summary"
    )
    assert len(summary.content) < SUMMARY_MAX_CHARS + 200
