"""Searcher.suggest_refinement + Orchestrator.extract_search_context (M3 T3.2.3.1, T3.2.3.2).

These tests pin the contract:
- Orchestrator builds a compact transcript snippet from working memory
  (most-recent-N messages, role-prefixed, cropped by chars).
- Searcher parses LLM JSON into a SearchSuggestion with sane defaults
  on bad / partial / empty output.
- Mode validation: known literals pass through, "group:<author>" is
  normalised (trimmed), unknown values become None (never break /search).

End-to-end /refine slash interaction (input loop + cmd_search dispatch)
lives in tests/unit/test_chat_refine_slash.py.
"""

from __future__ import annotations

import json

import pytest

from research_agent.agents.orchestrator import Orchestrator
from research_agent.agents.searcher import Searcher, SearchSuggestion
from research_agent.core.llm import MockLLMProvider
from research_agent.memory.working_memory import WorkingMemory

# ---------------------------------------------------------------- Searcher


def _make_searcher(payload: dict) -> Searcher:
    return Searcher(MockLLMProvider([json.dumps(payload)]))


def test_suggest_refinement_returns_empty_for_empty_context() -> None:
    searcher = _make_searcher({"query": "anything"})
    out = searcher.suggest_refinement("   ")
    assert out == SearchSuggestion(query="", mode=None, reason="", confidence=0.0)


def test_suggest_refinement_parses_full_payload() -> None:
    searcher = _make_searcher(
        {
            "query": "FlashAttention KV cache",
            "mode": "applied",
            "reason": "The critic flagged memory bandwidth as the bottleneck.",
            "confidence": 0.75,
        }
    )
    out = searcher.suggest_refinement(
        "analyst: paper proposes a new attention mechanism\n"
        "critic: but memory bandwidth is the bottleneck",
        previous_query="efficient attention",
    )
    assert out.query == "FlashAttention KV cache"
    assert out.mode == "applied"
    assert out.confidence == pytest.approx(0.75)
    assert "memory bandwidth" in out.reason


def test_suggest_refinement_drops_unknown_mode() -> None:
    searcher = _make_searcher(
        {"query": "x", "mode": "deeplearning", "reason": "r", "confidence": 0.5}
    )
    out = searcher.suggest_refinement("user: hello")
    assert out.mode is None


def test_suggest_refinement_accepts_group_mode() -> None:
    searcher = _make_searcher(
        {
            "query": "x",
            "mode": "group: Andrej Karpathy ",
            "reason": "r",
            "confidence": 0.4,
        }
    )
    out = searcher.suggest_refinement("user: hi")
    assert out.mode == "group:Andrej Karpathy"


def test_suggest_refinement_drops_group_mode_without_author() -> None:
    searcher = _make_searcher(
        {"query": "x", "mode": "group:", "reason": "r", "confidence": 0.4}
    )
    out = searcher.suggest_refinement("user: hi")
    assert out.mode is None


def test_suggest_refinement_clamps_confidence() -> None:
    searcher = _make_searcher(
        {"query": "x", "mode": None, "reason": "r", "confidence": 5.0}
    )
    out = searcher.suggest_refinement("user: hi")
    assert out.confidence == 1.0


def test_suggest_refinement_handles_garbage_json() -> None:
    # MockLLMProvider returns the string verbatim; if it isn't parseable
    # extract_json raises ValueError and the helper should degrade gracefully.
    searcher = Searcher(MockLLMProvider(["this is not json at all"]))
    out = searcher.suggest_refinement("user: hi")
    assert out == SearchSuggestion(query="", mode=None, reason="", confidence=0.0)


def test_suggest_refinement_handles_partial_payload() -> None:
    # Missing fields default to "" / None / 0.0 rather than crashing.
    searcher = _make_searcher({"query": "just a query"})
    out = searcher.suggest_refinement("user: hi")
    assert out.query == "just a query"
    assert out.mode is None
    assert out.reason == ""
    assert out.confidence == 0.0


# ----------------------------------------------- Orchestrator.extract_context


def _make_orchestrator() -> Orchestrator:
    return Orchestrator(MockLLMProvider([]))


def test_extract_search_context_empty_memory() -> None:
    orch = _make_orchestrator()
    memory = WorkingMemory.new_session()
    assert orch.extract_search_context(memory) == ""


def test_extract_search_context_skips_empty_messages() -> None:
    orch = _make_orchestrator()
    memory = WorkingMemory.new_session()
    memory.append("user", "")
    memory.append("user", "   ")
    assert orch.extract_search_context(memory) == ""


def test_extract_search_context_includes_recent_messages() -> None:
    orch = _make_orchestrator()
    memory = WorkingMemory.new_session()
    memory.append("user", "discuss efficient attention")
    memory.append("analyst", "The paper proposes a sparse pattern.")
    memory.append("critic", "But memory bandwidth is the bottleneck.")
    text = orch.extract_search_context(memory)
    assert "user: discuss efficient attention" in text
    assert "analyst: The paper proposes a sparse pattern." in text
    assert "critic: But memory bandwidth is the bottleneck." in text


def test_extract_search_context_caps_message_count() -> None:
    orch = _make_orchestrator()
    memory = WorkingMemory.new_session()
    for i in range(30):
        memory.append("user", f"msg-{i}")
    text = orch.extract_search_context(memory, max_messages=5)
    # Only the last 5 lines should be present.
    assert "msg-25" in text
    assert "msg-29" in text
    assert "msg-24" not in text


def test_extract_search_context_truncates_long_messages() -> None:
    orch = _make_orchestrator()
    memory = WorkingMemory.new_session()
    memory.append("analyst", "x" * 1200)
    text = orch.extract_search_context(memory)
    # Per-message cap is 600 chars + ellipsis.
    assert "…" in text
    # Find the analyst line and check length is roughly bounded.
    line = next(line for line in text.splitlines() if line.startswith("analyst:"))
    assert len(line) < 800


def test_extract_search_context_caps_total_chars() -> None:
    orch = _make_orchestrator()
    memory = WorkingMemory.new_session()
    for i in range(20):
        # Each message ~80 chars after role prefix.
        memory.append("user", f"message {i:02d}: " + "y" * 60)
    text = orch.extract_search_context(memory, max_messages=20, max_chars=400)
    assert len(text) <= 410
    # Tail must be preserved (most recent index 19 visible) - leading "…\n"
    # marker indicates truncation.
    assert text.startswith("…\n")
    assert "message 19" in text
