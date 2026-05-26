"""ChatSession — live state for one REPL run.

Holds backend handles (DB + repos + orchestrator + memory) plus the
conversation-specific anchor paper, current idea, and debate history so
slash handlers and the LLM agent loop can share one consistent state.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console

from research_agent.agents.debate import DebateHistory
from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.agents.orchestrator import Orchestrator
from research_agent.config import Config
from research_agent.core.llm import LLMProvider
from research_agent.core.paper import Paper
from research_agent.memory.working_memory import WorkingMemory
from research_agent.storage.database import Database, PaperRepository
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
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
    vectors: IdeaVectorStore
    keeper: MemoryKeeper
    orch: Orchestrator
    memory: WorkingMemory

    anchor_paper: Paper | None = None
    current_idea_id: str | None = None
    idea_seed: str = ""
    debate: DebateHistory = field(default_factory=DebateHistory)
    max_context_tokens: int = 8000

    @classmethod
    def create(
        cls,
        cfg: Config,
        llm: LLMProvider,
        console: Console,
        *,
        input_fn: Callable[[str], str] | None = None,
        use_chroma: bool | None = None,
    ) -> ChatSession:
        if input_fn is None:
            input_fn = console.input
        db = Database(cfg.db_path)
        discussions = DiscussionRepository(db)
        papers = PaperRepository(db)
        ideas = IdeaRepository(db)
        searches = SearchRepository(db)
        if use_chroma is None:
            use_chroma = not bool(os.environ.get("RESEARCH_AGENT_TEST_MODE"))
        vectors = IdeaVectorStore(cfg.chroma_dir, use_chroma=use_chroma)
        keeper = MemoryKeeper(ideas, vectors)
        orch = Orchestrator(llm, language=cfg.language)
        memory = WorkingMemory.new_session()
        return cls(
            cfg=cfg,
            llm=llm,
            console=console,
            input_fn=input_fn,
            db=db,
            discussions=discussions,
            papers=papers,
            ideas=ideas,
            searches=searches,
            vectors=vectors,
            keeper=keeper,
            orch=orch,
            memory=memory,
        )

    def set_anchor(self, paper: Paper) -> None:
        """Persist + register a paper as the current debate anchor."""
        self.papers.save(paper)
        self.anchor_paper = paper

    def close(self) -> int:
        """Flush memory to SQLite and close the DB. Returns messages persisted."""
        saved = self.memory.persist(self.discussions)
        self.db.close()
        return saved
