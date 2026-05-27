"""ChatSession — live state for one REPL run.

Holds backend handles (DB + repos + orchestrator + memory) plus the
conversation-specific anchor paper, current idea, and debate history so
slash handlers and the LLM agent loop can share one consistent state.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from research_agent.agents.debate import DebateHistory
from research_agent.agents.illustrator import FigureDraft
from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.agents.orchestrator import Orchestrator
from research_agent.agents.scribe import Draft
from research_agent.agents.searcher import Searcher
from research_agent.config import Config
from research_agent.core.llm import LLMProvider
from research_agent.core.paper import Paper
from research_agent.memory.working_memory import WorkingMemory
from research_agent.storage.database import Database, PaperRepository
from research_agent.storage.discussion_vectors import DiscussionVectorStore
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.reading_queue import ReadingQueueRepository
from research_agent.storage.searches import SearchRepository
from research_agent.storage.vector_store import IdeaVectorStore


@dataclass
class ChatSession:
    """Backend bundle + conversation state for a single REPL session."""

    cfg: Config
    llm: LLMProvider
    console: Console
    input_fn: Callable[[str], str]

    db: Database
    discussions: DiscussionRepository
    papers: PaperRepository
    ideas: IdeaRepository
    searches: SearchRepository
    queue: ReadingQueueRepository
    vectors: IdeaVectorStore
    discussion_vectors: DiscussionVectorStore
    keeper: MemoryKeeper
    orch: Orchestrator
    searcher: Searcher
    memory: WorkingMemory

    # Production REPL sets this to a prompt_toolkit ``PromptSession`` (built
    # by ``chat.prompt_ui.build_prompt_session``). Tests and headless flows
    # leave it ``None`` and rely on ``input_fn`` instead. Typed ``Any`` to
    # avoid leaking the optional dependency into this module's surface.
    prompt_session: Any | None = None

    anchor_paper: Paper | None = None
    current_idea_id: str | None = None
    idea_seed: str = ""
    # Last `/search` query (or LLM `search_arxiv` query) executed this
    # session - used by `Searcher.suggest_refinement` to anchor pivots.
    last_search_query: str = ""
    debate: DebateHistory = field(default_factory=DebateHistory)
    max_context_tokens: int = 8000

    # Writing-pipeline caches populated by draft_section / draft_figure /
    # revise_draft tools (Phase 2). Keyed by canonical section / figure
    # type so the LLM can resolve "the latest intro" without remembering
    # version ids. Saving to disk is a separate, explicit tool call so
    # nothing here ever hits the filesystem unless the user asks.
    recent_drafts: dict[str, list[Draft]] = field(default_factory=dict)
    recent_figures: dict[str, list[FigureDraft]] = field(default_factory=dict)
    recent_revisions: dict[str, Draft] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        cfg: Config,
        llm: LLMProvider,
        console: Console,
        *,
        input_fn: Callable[[str], str] | None = None,
        use_chroma: bool | None = None,
        prompt_session: Any | None = None,
    ) -> ChatSession:
        if input_fn is None:
            input_fn = console.input
        db = Database(cfg.db_path)
        discussions = DiscussionRepository(db)
        papers = PaperRepository(db)
        ideas = IdeaRepository(db)
        searches = SearchRepository(db)
        queue = ReadingQueueRepository(db)
        if use_chroma is None:
            use_chroma = not bool(os.environ.get("RESEARCH_AGENT_TEST_MODE"))
        vectors = IdeaVectorStore(cfg.chroma_dir, use_chroma=use_chroma)
        discussion_vectors = DiscussionVectorStore(
            cfg.chroma_dir, use_chroma=use_chroma
        )
        keeper = MemoryKeeper(
            ideas,
            vectors,
            discussions=discussions,
            discussion_vectors=discussion_vectors,
        )
        orch = Orchestrator(llm, language=cfg.language)
        searcher = Searcher(llm, language=cfg.language)
        memory = WorkingMemory.new_session()
        return cls(
            cfg=cfg,
            llm=llm,
            console=console,
            input_fn=input_fn,
            prompt_session=prompt_session,
            db=db,
            discussions=discussions,
            papers=papers,
            ideas=ideas,
            searches=searches,
            queue=queue,
            vectors=vectors,
            discussion_vectors=discussion_vectors,
            keeper=keeper,
            orch=orch,
            searcher=searcher,
            memory=memory,
        )

    def set_anchor(self, paper: Paper) -> None:
        """Persist + register a paper as the current debate anchor."""
        self.papers.save(paper)
        self.anchor_paper = paper

    def close(self) -> int:
        """Flush memory to SQLite + vector index and close the DB.

        Returns the number of messages newly persisted this session.
        """
        idea_id = self.memory.idea_id

        def _index(message_id: str, msg) -> None:  # type: ignore[no-untyped-def]
            self.keeper.index_message(
                message_id=message_id,
                session_id=self.memory.session_id,
                role=msg.role,
                content=msg.content,
                idea_id=idea_id,
            )

        saved = self.memory.persist(self.discussions, indexer=_index)
        self.db.close()
        return saved
