"""CLI handler for ``research write <section>``.

Drives the :class:`Scribe` agent: load the fingerprint, optionally
pull related material from the user's memory store, fan out N
parallel drafts, render them as a Rich panel-per-draft, and
optionally write the bouquet to a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.agents.scribe import Draft, Scribe, normalize_section
from research_agent.config import Config
from research_agent.core.idea import Idea
from research_agent.core.llm import LLMClient
from research_agent.storage.database import Database
from research_agent.storage.discussion_vectors import DiscussionVectorStore
from research_agent.storage.discussions import DiscussionMessage, DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore
from research_agent.style.fingerprint import Fingerprint

# Looser threshold than the parked-idea alert (0.8) - here we *want*
# any meaningful overlap; the LLM downstream will decide what's
# actually useful.
DEFAULT_CONTEXT_THRESHOLD = 0.5
DEFAULT_IDEA_LIMIT = 3
DEFAULT_DISCUSSION_LIMIT = 3
MAX_DRAFT_EXCERPT_CHARS = 2000


@dataclass
class WritingContext:
    """Everything we want the Scribe to know besides the fingerprint."""

    user_text: str = ""
    related_ideas: list[Idea] = field(default_factory=list)
    recent_discussions: list[DiscussionMessage] = field(default_factory=list)
    existing_drafts: list[tuple[str, str]] = field(default_factory=list)  # (label, body)

    def is_empty(self) -> bool:
        return (
            not self.user_text.strip()
            and not self.related_ideas
            and not self.recent_discussions
            and not self.existing_drafts
        )

    def render(self) -> str:
        """Flatten into a single string injected as ``Scribe.generate(context=...)``."""
        if self.is_empty():
            return ""
        parts: list[str] = []
        if self.user_text.strip():
            parts.append("[USER CONTEXT]")
            parts.append(self.user_text.strip())
        if self.related_ideas:
            parts.append("")
            parts.append("[RELATED IDEAS FROM YOUR LIBRARY]")
            for idea in self.related_ideas:
                score = (
                    f"{idea.critic_score:.0f}/9"
                    if idea.critic_score is not None
                    else "—"
                )
                summary = (idea.description or "").strip().replace("\n", " ")
                if len(summary) > 240:
                    summary = summary[:237] + "…"
                parts.append(
                    f"- {idea.title} ({idea.status}, last score {score}): {summary}"
                )
        if self.recent_discussions:
            parts.append("")
            parts.append("[RECENT DISCUSSION EXCERPTS]")
            for msg in self.recent_discussions:
                role = msg.role or "?"
                content = (msg.content or "").strip().replace("\n", " ")
                if len(content) > 280:
                    content = content[:277] + "…"
                parts.append(f"- [{role}] {content}")
        if self.existing_drafts:
            parts.append("")
            parts.append(
                "[EXISTING DRAFTS TO STAY CONSISTENT WITH — do not contradict, "
                "do not duplicate verbatim]"
            )
            for label, body in self.existing_drafts:
                excerpt = body.strip()
                if len(excerpt) > MAX_DRAFT_EXCERPT_CHARS:
                    excerpt = excerpt[:MAX_DRAFT_EXCERPT_CHARS] + "\n[... truncated ...]"
                parts.append(f"--- {label} ---")
                parts.append(excerpt)
        return "\n".join(parts)


@dataclass
class WriteResult:
    section: str
    drafts: list[Draft]
    context: WritingContext | None = None


def _build_writing_context(
    *,
    user_text: str,
    check_against: list[Path] | None,
    memory_keeper: MemoryKeeper | None,
    idea_limit: int = DEFAULT_IDEA_LIMIT,
    discussion_limit: int = DEFAULT_DISCUSSION_LIMIT,
    threshold: float = DEFAULT_CONTEXT_THRESHOLD,
    console: Console | None = None,
) -> WritingContext:
    """Assemble a rich context blob for the Scribe.

    The MemoryKeeper is queried only when ``user_text`` is non-empty
    (we use the user's description as the seed for vector similarity).
    ``check_against`` paths are read verbatim (truncated) so the
    Scribe is told *not* to repeat or contradict them.
    """
    ctx = WritingContext(user_text=user_text or "")

    if user_text.strip() and memory_keeper is not None:
        try:
            associations = memory_keeper.check_associations(
                user_text,
                threshold=threshold,
                limit=idea_limit,
                statuses=(),
            )
            ctx.related_ideas = [a.idea for a in associations]
        except Exception as exc:
            if console is not None:
                console.print(f"[dim]· skipping idea recall: {exc}[/dim]")
        try:
            ctx.recent_discussions = memory_keeper.recall_history(
                user_text, limit=discussion_limit
            )
        except Exception as exc:
            if console is not None:
                console.print(f"[dim]· skipping discussion recall: {exc}[/dim]")

    for path in check_against or []:
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as exc:
            if console is not None:
                console.print(
                    f"[yellow]Could not read --check-against {path}:[/yellow] {exc}"
                )
            continue
        ctx.existing_drafts.append((path.name, body))

    return ctx


def _open_memory_keeper(cfg: Config) -> tuple[MemoryKeeper | None, Database | None]:
    """Best-effort construction of a MemoryKeeper.

    Returns ``(None, None)`` if either the SQLite or the vector store
    can't be opened - ``research write`` still works in those
    degraded environments, just without auto-pulled context.
    """
    try:
        db = Database(cfg.db_path)
        ideas = IdeaRepository(db)
        vectors = IdeaVectorStore(cfg.chroma_dir)
        discussions = DiscussionRepository(db)
        discussion_vectors = DiscussionVectorStore(cfg.chroma_dir)
        keeper = MemoryKeeper(
            ideas,
            vectors,
            discussions=discussions,
            discussion_vectors=discussion_vectors,
        )
        return keeper, db
    except Exception:
        return None, None


def run_write(
    cfg: Config,
    console: Console,
    *,
    section: str,
    context: str = "",
    check_against: list[Path] | None = None,
    target_words: int = 300,
    versions: int = 3,
    output: Path | None = None,
    parallel: bool = True,
    scribe: Scribe | None = None,
    memory_keeper: MemoryKeeper | None = None,
) -> WriteResult:
    """Generate ``versions`` drafts of ``section`` and render them.

    ``scribe`` and ``memory_keeper`` are injectable for tests; in
    production we open them from ``cfg`` ourselves and close any
    SQLite handle we opened on exit.
    """
    canonical = normalize_section(section)

    fingerprint: Fingerprint | None = None
    if cfg.fingerprint_path.exists():
        try:
            fingerprint = Fingerprint.load_from(cfg.fingerprint_path)
        except Exception as exc:
            console.print(
                f"[yellow]Could not load fingerprint:[/yellow] {exc} "
                "— falling back to generic academic style."
            )
            fingerprint = None
    else:
        console.print(
            "[yellow]No style fingerprint at[/yellow] "
            f"{cfg.fingerprint_path}.\n"
            "Run [bold]research style train[/bold] + "
            "[bold]research style fingerprint[/bold] first for a voice match."
        )

    db_owned: Database | None = None
    if memory_keeper is None and (context.strip() or check_against):
        memory_keeper, db_owned = _open_memory_keeper(cfg)

    try:
        writing_ctx = _build_writing_context(
            user_text=context,
            check_against=check_against,
            memory_keeper=memory_keeper,
            console=console,
        )
        _render_context_preview(console, writing_ctx)
        rendered_ctx = writing_ctx.render()

        if scribe is None:
            llm = LLMClient.from_config(cfg)
            scribe = Scribe(llm, language=cfg.language)

        drafts = scribe.generate(
            canonical,
            fingerprint=fingerprint,
            context=rendered_ctx,
            target_words=target_words,
            n=versions,
            parallel=parallel,
        )

        _render_drafts(console, canonical, drafts)
        if output is not None:
            _persist_drafts(output, canonical, drafts)
            console.print(f"[green]✓[/green] Wrote drafts to [bold]{output}[/bold]")

        return WriteResult(section=canonical, drafts=drafts, context=writing_ctx)
    finally:
        if db_owned is not None:
            db_owned.close()


def _render_context_preview(console: Console, ctx: WritingContext) -> None:
    """Print a one-line summary of what context the Scribe will see."""
    if ctx.is_empty():
        return
    bits: list[str] = []
    if ctx.user_text.strip():
        bits.append("user context")
    if ctx.related_ideas:
        bits.append(f"{len(ctx.related_ideas)} related idea(s)")
    if ctx.recent_discussions:
        bits.append(f"{len(ctx.recent_discussions)} discussion excerpt(s)")
    if ctx.existing_drafts:
        bits.append(f"{len(ctx.existing_drafts)} draft(s) to stay consistent with")
    console.print(f"[dim]Scribe context:[/dim] {', '.join(bits)}")


def _render_drafts(console: Console, section: str, drafts: list[Draft]) -> None:
    if not drafts:
        console.print(f"[yellow]Scribe produced no drafts for {section}.[/yellow]")
        return
    for d in drafts:
        header = (
            f"[bold]Version {d.version}[/bold] · {d.variant_label} · "
            f"~{d.word_count} words (target {d.target_words})"
        )
        body = d.text or "[dim](empty)[/dim]"
        if d.style_note:
            body = f"[italic dim]{d.style_note}[/italic dim]\n\n{body}"
        console.print(Panel(body, title=header, border_style="cyan"))


def _persist_drafts(path: Path, section: str, drafts: list[Draft]) -> None:
    """Write drafts to a Markdown file, one section per draft."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Scribe drafts — {section}\n"]
    for d in drafts:
        lines.append(f"## Version {d.version} — {d.variant_label}\n")
        if d.style_note:
            lines.append(f"_{d.style_note}_\n")
        lines.append(f"*~{d.word_count} words (target {d.target_words})*\n")
        lines.append("")
        lines.append(d.text or "")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
