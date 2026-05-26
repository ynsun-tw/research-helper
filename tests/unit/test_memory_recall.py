"""Tests for cross-session discussion recall (M3 task 2)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich.console import Console

from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import LLM_TOOLS, cmd_recall, exec_recall_history
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall
from research_agent.memory.working_memory import WorkingMemory
from research_agent.storage.database import Database
from research_agent.storage.discussion_vectors import (
    INDEXABLE_ROLES,
    DiscussionVectorStore,
)
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore


def _make_session(tmp_path: Path) -> tuple[ChatSession, Console]:
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=140)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    return session, console


# --------------------------- DiscussionVectorStore ---------------------------


def test_indexable_roles_excludes_noise() -> None:
    assert "user" in INDEXABLE_ROLES
    assert "analyst" in INDEXABLE_ROLES
    assert "critic" in INDEXABLE_ROLES
    assert "system" not in INDEXABLE_ROLES
    assert "tool" not in INDEXABLE_ROLES


def test_vector_store_fallback_jaccard_returns_best_match(tmp_path: Path) -> None:
    store = DiscussionVectorStore(tmp_path / "chroma", use_chroma=False)
    store.upsert(
        message_id="m1",
        session_id="s1",
        role="user",
        content="positional encodings in transformers help with order",
    )
    store.upsert(
        message_id="m2",
        session_id="s1",
        role="analyst",
        content="diffusion models denoise iteratively from noise",
    )
    hits = store.query_similar("transformer positional encoding")
    assert hits[0] == "m1"


def test_vector_store_excludes_current_session(tmp_path: Path) -> None:
    store = DiscussionVectorStore(tmp_path / "chroma", use_chroma=False)
    store.upsert(
        message_id="m1",
        session_id="current",
        role="user",
        content="transformer attention layer scaling",
    )
    store.upsert(
        message_id="m2",
        session_id="past",
        role="user",
        content="transformer attention layer scaling",
    )
    hits = store.query_similar(
        "transformer attention scaling", exclude_session_id="current"
    )
    assert hits == ["m2"]


def test_vector_store_skips_empty_content(tmp_path: Path) -> None:
    store = DiscussionVectorStore(tmp_path / "chroma", use_chroma=False)
    store.upsert(message_id="m1", session_id="s1", role="user", content="   ")
    assert store.query_similar("anything") == []


# ----------------------------- MemoryKeeper recall ---------------------------


def test_recall_history_returns_empty_when_no_vectors(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    try:
        keeper = MemoryKeeper(
            IdeaRepository(db),
            IdeaVectorStore(tmp_path / "chroma", use_chroma=False),
            discussions=DiscussionRepository(db),
            discussion_vectors=None,
        )
        assert keeper.recall_history("anything") == []
    finally:
        db.close()


def test_recall_history_preserves_similarity_order(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    try:
        discussions = DiscussionRepository(db)
        # Append three past messages directly so we have stable ids.
        ids = []
        for role, content in [
            ("user", "transformer self-attention details"),
            ("analyst", "diffusion sampling steps explained"),
            ("user", "self-attention quadratic cost"),
        ]:
            ids.append(discussions.append("past", role, content))

        vectors = DiscussionVectorStore(tmp_path / "chroma", use_chroma=False)
        for mid, role, content in [
            (ids[0], "user", "transformer self-attention details"),
            (ids[1], "analyst", "diffusion sampling steps explained"),
            (ids[2], "user", "self-attention quadratic cost"),
        ]:
            vectors.upsert(
                message_id=mid, session_id="past", role=role, content=content
            )

        keeper = MemoryKeeper(
            IdeaRepository(db),
            IdeaVectorStore(tmp_path / "chroma", use_chroma=False),
            discussions=discussions,
            discussion_vectors=vectors,
        )
        matches = keeper.recall_history("self-attention", limit=3)
        contents = [m.content for m in matches]
        # Both self-attention messages should rank above the diffusion one.
        sa_messages = [c for c in contents if "self-attention" in c]
        assert len(sa_messages) >= 1
        if "diffusion sampling steps explained" in contents:
            assert contents.index(sa_messages[0]) < contents.index(
                "diffusion sampling steps explained"
            )
    finally:
        db.close()


# -------------------------- persist → index bridge ---------------------------


def test_working_memory_persist_invokes_indexer(tmp_path: Path) -> None:
    db = Database(tmp_path / "memory.db")
    try:
        repo = DiscussionRepository(db)
        memory = WorkingMemory.new_session("sess-1")
        memory.append("user", "transformer paper question")
        memory.append("analyst", "the model attends across positions")
        memory.append("system", "tool result noise that we should skip")

        seen: list[tuple[str, str, str]] = []

        def indexer(mid: str, msg) -> None:
            seen.append((mid, msg.role, msg.content))

        saved = memory.persist(repo, indexer=indexer)
        assert saved == 3
        assert len(seen) == 3
        # second persist should be a no-op (idempotency)
        assert memory.persist(repo, indexer=indexer) == 0
    finally:
        db.close()


def test_chat_session_close_indexes_user_and_agent_only(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    session.memory.append("user", "transformer attention details")
    session.memory.append("analyst", "self-attention compares all token pairs")
    session.memory.append("system", "noise: spinner output")
    saved = session.close()
    assert saved == 3
    # Only user + analyst rows should land in the vector store fallback.
    fallback = session.discussion_vectors._fallback  # type: ignore[attr-defined]
    roles = {role for role, _, _ in fallback.values()}
    assert roles == {"user", "analyst"}
    assert len(fallback) == 2


# ------------------------------ slash + LLM tool -----------------------------


def test_cmd_recall_prints_matches(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    # Seed the vector store with a message tagged to a *different* session.
    msg_id = session.discussions.append(
        "past-session", "user", "transformer self-attention scaling"
    )
    session.discussion_vectors.upsert(
        message_id=msg_id,
        session_id="past-session",
        role="user",
        content="transformer self-attention scaling",
    )
    cmd_recall(session, "self-attention transformer")
    out = console.file.getvalue()
    assert "Recalled" in out
    assert "self-attention" in out
    session.close()


def test_cmd_recall_empty_query_warns(tmp_path: Path) -> None:
    session, console = _make_session(tmp_path)
    cmd_recall(session, "")
    assert "Usage" in console.file.getvalue()
    session.close()


def test_recall_history_tool_registered_with_required_query() -> None:
    tool = LLM_TOOLS["recall_history"]
    schema = tool.schema["function"]
    assert "query" in schema["parameters"]["properties"]
    assert "query" in schema["parameters"]["required"]


def test_recall_history_tool_validates_query(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    assert "Error" in exec_recall_history(session, {})
    assert "Error" in exec_recall_history(session, {"query": "   "})
    session.close()


def test_recall_history_tool_returns_no_match_text(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    out = exec_recall_history(session, {"query": "something never discussed"})
    assert "No prior discussion recalled" in out
    session.close()


def test_recall_history_tool_returns_match_snippets(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    mid = session.discussions.append(
        "past", "user", "we explored sparse attention variants"
    )
    session.discussion_vectors.upsert(
        message_id=mid,
        session_id="past",
        role="user",
        content="we explored sparse attention variants",
    )
    text = exec_recall_history(session, {"query": "sparse attention"})
    assert "sparse attention" in text
    assert "session=past" in text
    session.close()


def test_recall_history_tool_clamps_limit(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path)
    assert "must be an integer" in exec_recall_history(
        session, {"query": "x", "limit": "abc"}
    )
    session.close()


# ----------------------- end-to-end chat agent loop --------------------------


def test_llm_chains_recall_history_via_chat_loop(tmp_path: Path) -> None:
    """End-to-end: model emits recall_history → router executes it → LLM
    composes a summary using the recalled snippet.

    Seeds the session's own vector store + discussions repo (the chat loop
    creates fresh instances; fallback-mode stores are in-memory and don't
    share state across instances, so we prepare the session first).
    """
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=140)
    llm = MockLLMProvider(
        [
            ChatResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="t1",
                        name="recall_history",
                        arguments=json.dumps(
                            {"query": "positional encodings long sequence"}
                        ),
                    ),
                ),
            ),
            ChatResponse(
                content="From last session we discussed positional encodings.",
                tool_calls=(),
            ),
        ]
    )
    session = ChatSession.create(
        cfg=cfg,
        llm=llm,
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    # Seed a past message into THIS session's stores so the in-memory
    # fallback vector index can see it.
    msg_id = session.discussions.append(
        "past-session-id",
        "user",
        "We talked about positional encodings for very long sequences",
    )
    session.discussion_vectors.upsert(
        message_id=msg_id,
        session_id="past-session-id",
        role="user",
        content="We talked about positional encodings for very long sequences",
    )

    from research_agent.chat.router import _chat_with_llm

    _chat_with_llm(session, "what did we say about positional encodings?")
    session.close()

    out = console.file.getvalue()
    # Executor printed the recall snippet …
    assert "We talked about positional" in out
    # … and the LLM composed a follow-up text using the recall result.
    assert "From last session we discussed positional encodings." in out
