"""Tool registry: slash command handlers + LLM-callable executors.

Each tool has a slash interface (``/name args`` for explicit invocation) and
optionally an LLM tool-calling interface (JSON-Schema parameters + executor
that returns a text result for the LLM's next reasoning step).

``SLASH_COMMANDS`` is consumed by router.py for slash dispatch and ``/help``.
``LLM_TOOLS`` exposes OpenAI-compatible function schemas to the agent loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from rich.table import Table

from research_agent.agents.analyst import AnalysisResult
from research_agent.agents.critic import CritiqueResult
from research_agent.agents.debate import DebateHistory, DebateResult, FollowUpResult
from research_agent.chat.session import ChatSession
from research_agent.cli_ideas import run_ideas_list, run_ideas_show, run_ideas_update
from research_agent.cli_services import _handle_debate_turn, _load_anchor_paper
from research_agent.core.idea import IDEA_STATUSES, IdeaStatus
from research_agent.core.llm import LLMError
from research_agent.core.loader import PaperLoadError
from research_agent.core.paper_resolver import search_arxiv_papers
from research_agent.search.arxiv_search import ArxivSearchError, ArxivSearchHit
from research_agent.search.semantic_scholar import SemanticScholarSearcher
from research_agent.storage.discussions import DiscussionMessage
from research_agent.storage.reading_queue import ALLOWED_STATUSES, QueueEntry, QueueStatus
from research_agent.storage.searches import StoredSearchQuery
from research_agent.ui.formatting import render_paper_header, render_read_report

SlashHandler = Callable[[ChatSession, str], None]
LLMExecutor = Callable[[ChatSession, dict[str, Any]], str]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    handler: SlashHandler
    summary: str
    usage: str


@dataclass(frozen=True, slots=True)
class LLMTool:
    name: str
    schema: dict[str, Any]
    executor: LLMExecutor


SLASH_COMMANDS: dict[str, SlashCommand] = {}
LLM_TOOLS: dict[str, LLMTool] = {}


def slash(
    name: str,
    *,
    summary: str,
    usage: str,
) -> Callable[[SlashHandler], SlashHandler]:
    """Register a slash command handler."""

    def deco(fn: SlashHandler) -> SlashHandler:
        SLASH_COMMANDS[name] = SlashCommand(
            name=name, handler=fn, summary=summary, usage=usage
        )
        return fn

    return deco


def _function_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def register_llm_tool(
    name: str, schema: dict[str, Any]
) -> Callable[[LLMExecutor], LLMExecutor]:
    """Register an LLM-callable executor; schema must use OpenAI function shape."""

    def deco(fn: LLMExecutor) -> LLMExecutor:
        LLM_TOOLS[name] = LLMTool(name=name, schema=schema, executor=fn)
        return fn

    return deco


def llm_tool_schemas() -> list[dict[str, Any]]:
    """Return all registered LLM tool schemas (OpenAI ``tools`` payload)."""
    return [t.schema for t in LLM_TOOLS.values()]


# ---------------------------------------------------------------- search


@slash(
    "search",
    summary="Search arXiv + LLM relevance score (logs to history; flags already-read).",
    usage="/search <keywords>",
)
def cmd_search(session: ChatSession, args: str) -> None:
    if not args:
        session.console.print("[yellow]Usage:[/yellow] /search <keywords>")
        return
    try:
        with session.console.status("[bold]Searching arXiv…[/bold]"):
            hits = search_arxiv_papers(args, max_results=5)
    except (PaperLoadError, ArxivSearchError) as exc:
        session.console.print(f"[red]Error:[/red] {exc}")
        return
    if not hits:
        session.console.print(f"[yellow]No results for[/yellow] {args!r}.")
        session.searches.record(
            args, hits, source="arxiv", session_id=session.memory.session_id
        )
        session.memory.append("system", _search_summary(args, hits))
        return

    scored = _score_hits(session, args, hits)
    session.searches.record(
        args, scored, source="arxiv", session_id=session.memory.session_id
    )
    sorted_hits = _sort_by_score(scored)
    already_read = session.searches.already_read([h.arxiv_id for h in sorted_hits])
    _render_search_hits(session, args, sorted_hits, already_read=already_read)
    session.memory.append(
        "system",
        _search_summary(args, sorted_hits, already_read=already_read),
    )


def _score_hits(
    session: ChatSession,
    query: str,
    hits: list[ArxivSearchHit],
) -> list[ArxivSearchHit]:
    try:
        with session.console.status("[bold]Scoring relevance (LLM)…[/bold]"):
            return session.searcher.score_hits(query, hits)
    except LLMError as exc:
        session.console.print(
            f"[yellow]Relevance scoring failed:[/yellow] {exc}. "
            "Showing unscored arXiv order."
        )
        return list(hits)


def _sort_by_score(hits: list[ArxivSearchHit]) -> list[ArxivSearchHit]:
    if not any(h.relevance_score is not None for h in hits):
        return list(hits)
    return sorted(hits, key=lambda h: -(h.relevance_score or 0.0))


def _render_search_hits(
    session: ChatSession,
    query: str,
    hits: list[ArxivSearchHit],
    *,
    already_read: set[str] | None = None,
    title: str | None = None,
    show_fallback_note: bool = True,
) -> None:
    """Render hits as a Rich table.

    ``title`` overrides the auto-derived table title (used by /cites and
    /refs to label "Citations of …" / "References from …"). Callers that
    intentionally hit Semantic Scholar (citation graph) pass
    ``show_fallback_note=False`` so users don't see the misleading
    "arXiv fallback" footer.
    """
    already_read = already_read or set()
    show_score = any(h.relevance_score is not None for h in hits)
    table_title = title or f"{_source_label(hits)} results for: {query}"
    table = Table(title=table_title, show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Year", justify="right")
    if show_score:
        table.add_column("Score", justify="right")
        table.add_column("Why", style="dim")
    table.add_column("Read", justify="center")
    for hit in hits:
        year = hit.published[:4] if hit.published else "—"
        read_mark = "[green]✓[/green]" if hit.arxiv_id in already_read else ""
        if show_score:
            score_cell = (
                f"{hit.relevance_score:.2f}"
                if hit.relevance_score is not None
                else "—"
            )
            reason = hit.relevance_reason or ""
            if len(reason) > 60:
                reason = reason[:57] + "…"
            table.add_row(
                hit.arxiv_id,
                hit.title[:80],
                year,
                score_cell,
                reason,
                read_mark,
            )
        else:
            table.add_row(hit.arxiv_id, hit.title[:80], year, read_mark)
    session.console.print(table)
    extras = []
    if show_score:
        extras.append("[dim]Sorted by relevance (LLM, 0-1).[/dim]")
    if show_fallback_note and any(h.source != "arxiv" for h in hits):
        extras.append(
            "[dim yellow]Note:[/dim yellow] [dim]results came from "
            "Semantic Scholar (arXiv fallback).[/dim]"
        )
    extras.append(
        "[dim]Use[/dim] /read <id> [dim]to load;[/dim] "
        "[green]✓[/green] [dim]= already in your library.[/dim]"
    )
    session.console.print(" ".join(extras))


def _source_label(hits: list[ArxivSearchHit]) -> str:
    """Pretty label for the table title; honours fallback sources."""
    if not hits:
        return "arXiv"
    src = hits[0].source
    if src == "semantic_scholar":
        return "Semantic Scholar (arXiv fallback)"
    return "arXiv"


def _search_summary(
    query: str,
    hits: list[ArxivSearchHit],
    *,
    already_read: set[str] | None = None,
) -> str:
    if not hits:
        return f"Searched arXiv for {query!r}; no results."
    read_set = already_read or set()
    rows: list[str] = []
    for h in hits:
        marker = " [read]" if h.arxiv_id in read_set else ""
        score_str = (
            f" score={h.relevance_score:.2f}" if h.relevance_score is not None else ""
        )
        rows.append(f"- {h.arxiv_id}: {h.title[:80]}{score_str}{marker}")
    return f"Searched arXiv for {query!r}; {len(hits)} results:\n" + "\n".join(rows)


@register_llm_tool(
    "search_arxiv",
    _function_schema(
        "search_arxiv",
        "Search arXiv for papers matching a keyword query. Returns a list of "
        "candidate papers (id, title, year) but does not load them.",
        {
            "query": {"type": "string", "description": "Keywords or title fragments."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return (default 5, max 10).",
                "minimum": 1,
                "maximum": 10,
            },
        },
        required=["query"],
    ),
)
def exec_search_arxiv(session: ChatSession, args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: query is required."
    max_results = int(args.get("max_results", 5))
    max_results = max(1, min(max_results, 10))
    try:
        with session.console.status(f"[bold]Searching arXiv:[/bold] {query}…"):
            hits = search_arxiv_papers(query, max_results=max_results)
    except (PaperLoadError, ArxivSearchError) as exc:
        return f"Error: {exc}"
    if not hits:
        session.searches.record(
            query, hits, source="arxiv", session_id=session.memory.session_id
        )
        return _search_summary(query, hits)
    scored = _score_hits(session, query, hits)
    session.searches.record(
        query, scored, source="arxiv", session_id=session.memory.session_id
    )
    sorted_hits = _sort_by_score(scored)
    already_read = session.searches.already_read([h.arxiv_id for h in sorted_hits])
    _render_search_hits(session, query, sorted_hits, already_read=already_read)
    return _search_summary(query, sorted_hits, already_read=already_read)


# ---------------------------------------------------------------- history


@slash(
    "history",
    summary="Show recent /search queries (cross-session, with already-read marks).",
    usage="/history [N]",
)
def cmd_history(session: ChatSession, args: str) -> None:
    limit = 10
    if args:
        try:
            limit = max(1, min(int(args.strip()), 50))
        except ValueError:
            session.console.print(
                "[yellow]Usage:[/yellow] /history [N]   (N is a positive integer)"
            )
            return

    queries = session.searches.recent_queries(limit=limit)
    if not queries:
        session.console.print("[dim]No search history yet. Try /search <keywords>.[/dim]")
        return
    _render_history(session, queries)


def _render_history(session: ChatSession, queries: list[StoredSearchQuery]) -> None:
    table = Table(
        title=f"Recent searches (showing {len(queries)})", show_header=True
    )
    table.add_column("When", style="dim")
    table.add_column("Source", style="dim")
    table.add_column("Query", style="cyan")
    table.add_column("Hits", justify="right")
    table.add_column("Read", justify="right")
    for q in queries:
        read_count = sum(1 for h in q.hits if h.read)
        table.add_row(
            q.created_at,
            q.source,
            q.query,
            str(len(q.hits)),
            f"{read_count}/{len(q.hits)}" if q.hits else "0/0",
        )
    session.console.print(table)
    session.console.print(
        "[dim]Re-run a query with[/dim] /search <keywords>; "
        "[dim]load any hit with[/dim] /read <arxiv-id>."
    )


def _history_summary_text(queries: list[StoredSearchQuery]) -> str:
    """Plain-text rendering for the LLM tool result; arxiv_ids stay machine-readable."""
    if not queries:
        return "No search history yet."
    lines: list[str] = ["Recent searches (most recent first):"]
    for i, q in enumerate(queries, start=1):
        read_count = sum(1 for h in q.hits if h.read)
        lines.append(
            f"{i}. [{q.created_at}] source={q.source} "
            f'query="{q.query}" — {len(q.hits)} hit(s), '
            f"{read_count} already read"
        )
        for h in q.hits:
            year = h.published[:4] if h.published else "—"
            mark = " [READ]" if h.read else ""
            score = (
                f" score={h.relevance_score:.2f}"
                if h.relevance_score is not None
                else ""
            )
            title = h.title if len(h.title) <= 100 else h.title[:97] + "…"
            lines.append(f"   - {h.arxiv_id} ({year}){score} {title}{mark}")
    return "\n".join(lines)


@register_llm_tool(
    "recent_searches",
    _function_schema(
        "recent_searches",
        "List recent /search queries (across past sessions) along with their "
        "hits and which arXiv ids the user has already loaded. Use this when "
        "the user refers to a prior search (e.g. 'open the BERT paper from "
        "yesterday') to recover the right arxiv_id, then chain into load_paper.",
        {
            "limit": {
                "type": "integer",
                "description": "How many recent queries to return (default 10, max 25).",
                "minimum": 1,
                "maximum": 25,
            },
        },
    ),
)
def exec_recent_searches(session: ChatSession, args: dict[str, Any]) -> str:
    limit_raw = args.get("limit", 10)
    try:
        limit = max(1, min(int(limit_raw), 25))
    except (TypeError, ValueError):
        return "Error: limit must be an integer between 1 and 25."
    queries = session.searches.recent_queries(limit=limit)
    if queries:
        # Also render to the console so the user sees what the model is reading.
        _render_history(session, queries)
    return _history_summary_text(queries)


# --------------------------------------------------------- citation graph


def _resolve_citation_target(session: ChatSession, args: str) -> str | None:
    """Return an arxiv_id from ``args``, else the anchor paper, else None.

    Anchor papers carry an id like ``arxiv:1706.03762``; we strip the
    ``arxiv:`` prefix. Local PDFs (``local:<sha1>``) have no arXiv
    mapping so we can't query citations for them - return None there
    too, and the caller asks the user for an explicit id.
    """
    explicit = args.strip()
    if explicit:
        return explicit
    if session.anchor_paper is None:
        return None
    pid = session.anchor_paper.id
    if pid.startswith("arxiv:"):
        return pid[len("arxiv:"):]
    return None


def _citation_summary(
    arxiv_id: str,
    hits: list[ArxivSearchHit],
    *,
    relation: str,
    already_read: set[str] | None = None,
) -> str:
    """Plain-text summary for the LLM tool result / working memory note."""
    if not hits:
        return f"No {relation} found for arxiv:{arxiv_id}."
    read_set = already_read or set()
    rows: list[str] = [
        f"Found {len(hits)} {relation} for arxiv:{arxiv_id}:",
    ]
    for h in hits:
        year = h.published[:4] if h.published else "—"
        mark = " [read]" if h.arxiv_id in read_set else ""
        rows.append(f"- {h.arxiv_id} ({year}) {h.title[:100]}{mark}")
    return "\n".join(rows)


def _run_citation_lookup(
    session: ChatSession,
    arxiv_id: str,
    *,
    relation: str,
    max_results: int,
) -> tuple[list[ArxivSearchHit], set[str]] | None:
    """Shared S2 lookup; renders error to console and returns None on failure."""
    s2 = SemanticScholarSearcher()
    fetcher = s2.get_citations if relation == "citations" else s2.get_references
    label = "citations of" if relation == "citations" else "references from"
    try:
        with session.console.status(
            f"[bold]Fetching {label} arxiv:{arxiv_id}…[/bold]"
        ):
            hits = fetcher(arxiv_id, max_results=max_results)
    except ArxivSearchError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")
        return None
    already_read = session.searches.already_read([h.arxiv_id for h in hits])
    return hits, already_read


def _render_citations(
    session: ChatSession,
    arxiv_id: str,
    hits: list[ArxivSearchHit],
    *,
    relation: str,
    already_read: set[str],
) -> None:
    label = "Citations of" if relation == "citations" else "References from"
    if not hits:
        session.console.print(
            f"[yellow]No {relation} found for[/yellow] arxiv:{arxiv_id}."
        )
        return
    _render_search_hits(
        session,
        arxiv_id,
        hits,
        already_read=already_read,
        title=f"{label} arxiv:{arxiv_id} (via Semantic Scholar)",
        show_fallback_note=False,
    )
    session.console.print(
        "[dim]Add any to your queue with[/dim] /queue add <id>; "
        "[dim]or load now with[/dim] /read <id>."
    )


@slash(
    "cites",
    summary="Show papers that cite the anchor paper (forward references, via S2).",
    usage="/cites [arxiv-id]",
)
def cmd_cites(session: ChatSession, args: str) -> None:
    target = _resolve_citation_target(session, args)
    if target is None:
        session.console.print(
            "[yellow]Usage:[/yellow] /cites <arxiv-id>  "
            "[dim](or /read a paper first to set the anchor)[/dim]"
        )
        return
    result = _run_citation_lookup(session, target, relation="citations", max_results=10)
    if result is None:
        return
    hits, already_read = result
    _render_citations(
        session, target, hits, relation="citations", already_read=already_read
    )
    session.memory.append(
        "system",
        _citation_summary(
            target, hits, relation="citations", already_read=already_read
        ),
    )


@slash(
    "refs",
    summary="Show papers cited by the anchor paper (backward references, via S2).",
    usage="/refs [arxiv-id]",
)
def cmd_refs(session: ChatSession, args: str) -> None:
    target = _resolve_citation_target(session, args)
    if target is None:
        session.console.print(
            "[yellow]Usage:[/yellow] /refs <arxiv-id>  "
            "[dim](or /read a paper first to set the anchor)[/dim]"
        )
        return
    result = _run_citation_lookup(session, target, relation="references", max_results=10)
    if result is None:
        return
    hits, already_read = result
    _render_citations(
        session, target, hits, relation="references", already_read=already_read
    )
    session.memory.append(
        "system",
        _citation_summary(
            target, hits, relation="references", already_read=already_read
        ),
    )


@register_llm_tool(
    "get_citations",
    _function_schema(
        "get_citations",
        "List papers that cite the given arXiv paper (forward references, "
        "Semantic Scholar). Useful for finding follow-up work.",
        {
            "arxiv_id": {
                "type": "string",
                "description": "arXiv id of the paper whose citations you want.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many citations to return (default 10, max 25).",
                "minimum": 1,
                "maximum": 25,
            },
        },
        required=["arxiv_id"],
    ),
)
def exec_get_citations(session: ChatSession, args: dict[str, Any]) -> str:
    arxiv_id = str(args.get("arxiv_id", "")).strip()
    if not arxiv_id:
        return "Error: arxiv_id is required."
    max_results = max(1, min(int(args.get("max_results", 10)), 25))
    result = _run_citation_lookup(
        session, arxiv_id, relation="citations", max_results=max_results
    )
    if result is None:
        return f"Error: failed to fetch citations for {arxiv_id}."
    hits, already_read = result
    _render_citations(
        session, arxiv_id, hits, relation="citations", already_read=already_read
    )
    return _citation_summary(
        arxiv_id, hits, relation="citations", already_read=already_read
    )


@register_llm_tool(
    "get_references",
    _function_schema(
        "get_references",
        "List papers cited by the given arXiv paper (backward references, "
        "Semantic Scholar). Useful for tracing intellectual lineage.",
        {
            "arxiv_id": {
                "type": "string",
                "description": "arXiv id of the paper whose bibliography you want.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many references to return (default 10, max 25).",
                "minimum": 1,
                "maximum": 25,
            },
        },
        required=["arxiv_id"],
    ),
)
def exec_get_references(session: ChatSession, args: dict[str, Any]) -> str:
    arxiv_id = str(args.get("arxiv_id", "")).strip()
    if not arxiv_id:
        return "Error: arxiv_id is required."
    max_results = max(1, min(int(args.get("max_results", 10)), 25))
    result = _run_citation_lookup(
        session, arxiv_id, relation="references", max_results=max_results
    )
    if result is None:
        return f"Error: failed to fetch references for {arxiv_id}."
    hits, already_read = result
    _render_citations(
        session, arxiv_id, hits, relation="references", already_read=already_read
    )
    return _citation_summary(
        arxiv_id, hits, relation="references", already_read=already_read
    )


# --------------------------------------------------------------- recall


@slash(
    "recall",
    summary="Search past discussions across all sessions for similar content.",
    usage="/recall <query keywords>",
)
def cmd_recall(session: ChatSession, args: str) -> None:
    query = args.strip()
    if not query:
        session.console.print("[yellow]Usage:[/yellow] /recall <query keywords>")
        return
    matches = session.keeper.recall_history(
        query, limit=5, exclude_session_id=session.memory.session_id
    )
    if not matches:
        session.console.print(
            "[dim]Nothing recalled from past sessions. "
            "Discussions are indexed when a session ends — keep using "
            "/discuss and try again next session.[/dim]"
        )
        return
    _render_recall(session, matches, query=query)


def _render_recall(
    session: ChatSession, matches: list[DiscussionMessage], *, query: str
) -> None:
    session.console.print(
        f"[bold]Recalled {len(matches)} past message(s)[/bold] for: "
        f"[cyan]{query}[/cyan]"
    )
    for i, m in enumerate(matches, start=1):
        snippet = m.content.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "…"
        session.console.print(
            f"  [dim]{i}.[/dim] [yellow]{m.role}[/yellow] "
            f"[dim](session {m.session_id[:8]}…)[/dim] {snippet}"
        )


def _recall_text(matches: list[DiscussionMessage], *, query: str) -> str:
    if not matches:
        return f"No prior discussion recalled for query: {query!r}."
    lines = [f"Recalled {len(matches)} past message(s) for: {query!r}"]
    for i, m in enumerate(matches, start=1):
        snippet = m.content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:397] + "…"
        idea = m.metadata.get("idea_id") if isinstance(m.metadata, dict) else None
        idea_tag = f" idea_id={idea}" if idea else ""
        lines.append(
            f"{i}. role={m.role} session={m.session_id}{idea_tag}\n   {snippet}"
        )
    return "\n".join(lines)


@register_llm_tool(
    "recall_history",
    _function_schema(
        "recall_history",
        "Search the user's past discussions (across all REPL sessions) for "
        "messages semantically similar to ``query``. Use this when the user "
        "refers to something previously discussed (e.g. 'what did we say "
        "about positional encodings last time?'). Returns the top matches "
        "with role + a snippet; chain into list_ideas or load_paper if you "
        "need the underlying artifact.",
        {
            "query": {
                "type": "string",
                "description": "Natural-language description of the topic to recall.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of past messages to return (default 5, max 15).",
                "minimum": 1,
                "maximum": 15,
            },
        },
        required=["query"],
    ),
)
def exec_recall_history(session: ChatSession, args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: query is required."
    limit_raw = args.get("limit", 5)
    try:
        limit = max(1, min(int(limit_raw), 15))
    except (TypeError, ValueError):
        return "Error: limit must be an integer between 1 and 15."
    matches = session.keeper.recall_history(
        query, limit=limit, exclude_session_id=session.memory.session_id
    )
    if matches:
        _render_recall(session, matches, query=query)
    return _recall_text(matches, query=query)


# ---------------------------------------------------------------- queue


_QUEUE_SUBS_HELP = (
    "[yellow]Usage:[/yellow] "
    "/queue [list|add <id> [title…]|remove <id>|done <id>|skip <id>|read|next]"
)


@slash(
    "queue",
    summary="Manage the reading queue (add / list / remove / read next).",
    usage="/queue [list|add <id>|remove <id>|done <id>|skip <id>|read|next]",
)
def cmd_queue(session: ChatSession, args: str) -> None:
    parts = args.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in {"list", ""}:
        status_filter: QueueStatus | None = None
        if rest:
            r = rest.strip().lower()
            if r == "all":
                status_filter = None
            elif r in ALLOWED_STATUSES:
                status_filter = r  # type: ignore[assignment]
            else:
                session.console.print(
                    f"[yellow]Unknown filter[/yellow] {r!r}. "
                    f"Use one of: all, {', '.join(sorted(ALLOWED_STATUSES))}."
                )
                return
        else:
            status_filter = "pending"
        _render_queue(session, status_filter)
        return

    if sub == "add":
        if not rest:
            session.console.print(
                "[yellow]Usage:[/yellow] /queue add <arxiv-id> [title…]"
            )
            return
        bits = rest.split(maxsplit=1)
        arxiv_id = bits[0]
        title = bits[1] if len(bits) > 1 else ""
        entry = session.queue.add(arxiv_id, title=title, source="manual")
        session.console.print(
            f"[green]Queued[/green] {entry.arxiv_id}"
            + (f" - {entry.title}" if entry.title else "")
        )
        return

    if sub in {"remove", "rm"}:
        if not rest:
            session.console.print("[yellow]Usage:[/yellow] /queue remove <arxiv-id>")
            return
        removed = session.queue.remove(rest.strip())
        if removed:
            session.console.print(f"[green]Removed[/green] {rest.strip()} from queue.")
        else:
            session.console.print(f"[yellow]Not in queue:[/yellow] {rest.strip()}")
        return

    if sub in {"done", "skip"}:
        target_status: QueueStatus = "done" if sub == "done" else "skipped"
        if not rest:
            session.console.print(
                f"[yellow]Usage:[/yellow] /queue {sub} <arxiv-id>"
            )
            return
        updated: QueueEntry | None = session.queue.set_status(
            rest.strip(), target_status
        )
        if updated is None:
            session.console.print(f"[yellow]Not in queue:[/yellow] {rest.strip()}")
        else:
            session.console.print(
                f"[green]Marked[/green] {updated.arxiv_id} as {updated.status}."
            )
        return

    if sub == "next":
        next_entry: QueueEntry | None = session.queue.next_pending()
        if next_entry is None:
            session.console.print(
                "[dim]Queue is empty. Add papers with /queue add <id>.[/dim]"
            )
            return
        session.console.print(
            f"[bold]Next up:[/bold] [cyan]{next_entry.arxiv_id}[/cyan] "
            f"{next_entry.title}"
        )
        session.console.print("[dim]Run[/dim] /queue read [dim]to load it.[/dim]")
        return

    if sub == "read":
        read_entry: QueueEntry | None = session.queue.next_pending()
        if read_entry is None:
            session.console.print(
                "[dim]Queue is empty. Add papers with /queue add <id>.[/dim]"
            )
            return
        session.queue.set_status(read_entry.arxiv_id, "in_progress")
        session.console.print(
            f"[bold]Reading next pending:[/bold] "
            f"{read_entry.arxiv_id} {read_entry.title}"
        )
        result = _load_and_analyze(session, read_entry.arxiv_id)
        # status update is handled inside _load_and_analyze success path
        if result is not None:
            session.console.print(f"[dim]{result.splitlines()[0]}[/dim]")
        return

    session.console.print(_QUEUE_SUBS_HELP)


def _render_queue(
    session: ChatSession, status: QueueStatus | None
) -> None:
    entries = session.queue.list(status=status)
    if not entries:
        if status is None or status == "pending":
            session.console.print(
                "[dim]Reading queue is empty. Use[/dim] /queue add <arxiv-id> "
                "[dim]to add a paper.[/dim]"
            )
        else:
            session.console.print(
                f"[dim]No entries with status[/dim] {status}."
            )
        return
    title = (
        "Reading queue (pending)"
        if status == "pending"
        else "Reading queue"
        if status is None
        else f"Reading queue ({status})"
    )
    table = Table(title=title, show_header=True)
    table.add_column("arXiv ID", style="cyan")
    table.add_column("Title")
    table.add_column("Status", justify="center")
    table.add_column("Added", style="dim")
    for e in entries:
        table.add_row(e.arxiv_id, (e.title or "(no title)")[:80], e.status, e.added_at)
    session.console.print(table)


def _queue_summary_text(entries: list[QueueEntry], *, label: str) -> str:
    if not entries:
        return f"{label}: queue is empty."
    lines = [f"{label} ({len(entries)} entry/ies):"]
    for e in entries:
        title = e.title or "(no title)"
        if len(title) > 100:
            title = title[:97] + "…"
        lines.append(f"- {e.arxiv_id} [{e.status}] {title}")
    return "\n".join(lines)


@register_llm_tool(
    "queue_add",
    _function_schema(
        "queue_add",
        "Add an arXiv paper to the user's reading queue (status=pending). "
        "If the paper is already queued, refreshes its title/source instead.",
        {
            "arxiv_id": {
                "type": "string",
                "description": "arXiv id, e.g. 1706.03762.",
            },
            "title": {
                "type": "string",
                "description": "Optional title to store with the entry.",
            },
        },
        required=["arxiv_id"],
    ),
)
def exec_queue_add(session: ChatSession, args: dict[str, Any]) -> str:
    arxiv_id = str(args.get("arxiv_id", "")).strip()
    if not arxiv_id:
        return "Error: arxiv_id is required."
    title = str(args.get("title", "")).strip()
    entry = session.queue.add(arxiv_id, title=title, source="llm")
    session.console.print(
        f"[green]Queued[/green] {entry.arxiv_id}"
        + (f" - {entry.title}" if entry.title else "")
    )
    return (
        f"Queued {entry.arxiv_id} with status={entry.status}"
        + (f" (title: {entry.title})" if entry.title else "")
    )


@register_llm_tool(
    "queue_list",
    _function_schema(
        "queue_list",
        "List entries in the reading queue. Defaults to 'pending'; pass "
        "status='all' to include done/skipped too.",
        {
            "status": {
                "type": "string",
                "description": (
                    "One of pending, in_progress, done, skipped, or 'all' "
                    "for every status."
                ),
            },
        },
    ),
)
def exec_queue_list(session: ChatSession, args: dict[str, Any]) -> str:
    raw = str(args.get("status", "pending")).strip().lower() or "pending"
    if raw == "all":
        entries = session.queue.list()
        label = "Reading queue (all)"
    elif raw in ALLOWED_STATUSES:
        entries = session.queue.list(status=raw)  # type: ignore[arg-type]
        label = f"Reading queue ({raw})"
    else:
        return (
            f"Error: status must be one of all, {', '.join(sorted(ALLOWED_STATUSES))}."
        )
    if entries:
        _render_queue(session, raw if raw in ALLOWED_STATUSES else None)  # type: ignore[arg-type]
    return _queue_summary_text(entries, label=label)


@register_llm_tool(
    "queue_next",
    _function_schema(
        "queue_next",
        "Return the next pending paper in the user's reading queue (FIFO). "
        "Returns the arxiv_id so you can chain into load_paper. Does not "
        "mutate state.",
        {},
    ),
)
def exec_queue_next(session: ChatSession, args: dict[str, Any]) -> str:
    entry = session.queue.next_pending()
    if entry is None:
        return "Queue is empty."
    title = f" (title: {entry.title})" if entry.title else ""
    return f"Next pending: {entry.arxiv_id}{title}. Use load_paper to read it."


# ---------------------------------------------------------------- read


@slash(
    "read",
    summary="Load a paper and run Analyst + Critic; sets the anchor.",
    usage="/read <arxiv-id | title keywords | path.pdf>",
)
def cmd_read(session: ChatSession, args: str) -> None:
    if not args:
        session.console.print("[yellow]Usage:[/yellow] /read <arxiv-id | title | path.pdf>")
        return
    result = _load_and_analyze(session, args)
    if result is not None:
        session.console.print(f"[dim]{result.splitlines()[0]}[/dim]")


def _load_and_analyze(session: ChatSession, source: str) -> str | None:
    """Shared implementation for /read and load_paper LLM tool. Returns summary text."""
    try:
        paper = _load_anchor_paper(
            source,
            cfg=session.cfg,
            console=session.console,
            input_fn=session.input_fn,
        )
    except PaperLoadError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")
        return f"Error loading paper: {exc}"

    session.set_anchor(paper)
    render_paper_header(session.console, paper)

    try:
        with session.console.status("[bold]Analyzing (Analyst + Critic)…[/bold]"):
            report = asyncio.run(session.orch.analyze_paper_parallel(paper))
    except LLMError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")
        return f"Error analyzing paper: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        session.console.print(f"[red]Error:[/red] Could not parse model output: {exc}")
        return f"Error parsing analysis: {exc}"

    session.papers.save_analysis_notes(
        paper.id,
        analyst_notes=_analysis_to_dict(report.analyst),
        critic_notes=_critique_to_dict(report.critic),
    )
    render_read_report(session.console, paper, report)
    session.memory.append(
        "system",
        f"Loaded and analyzed paper: {paper.title} ({paper.id})",
    )
    if session.queue.get(paper.id) is not None:
        session.queue.set_status(paper.id, "done")
        session.console.print(
            f"[dim]✓ Marked[/dim] {paper.id} [dim]as done in the reading queue.[/dim]"
        )
    session.debate.rounds.clear()
    session.idea_seed = ""
    return (
        f"Loaded paper '{paper.title}' ({paper.id}). "
        f"Analyst contributions: {report.analyst.contributions}. "
        f"Critic objections: {report.critic.objections}. "
        f"Critic support score: {report.critic.support_score}/9."
    )


@register_llm_tool(
    "load_paper",
    _function_schema(
        "load_paper",
        "Download a paper (by arXiv id, title keywords, or local PDF path) and "
        "run Analyst + Critic on it. Sets the paper as the conversation anchor.",
        {
            "source": {
                "type": "string",
                "description": "arXiv id (e.g. 1706.03762), title keywords, or PDF path.",
            },
        },
        required=["source"],
    ),
)
def exec_load_paper(session: ChatSession, args: dict[str, Any]) -> str:
    source = str(args.get("source", "")).strip()
    if not source:
        return "Error: source is required."
    return _load_and_analyze(session, source) or "Loaded."


# ---------------------------------------------------------------- discuss


@slash(
    "discuss",
    summary="Debate an idea grounded in the anchor paper.",
    usage="/discuss <your idea or follow-up question>",
)
def cmd_discuss(session: ChatSession, args: str) -> None:
    if session.anchor_paper is None:
        session.console.print(
            "[yellow]No anchor paper loaded.[/yellow] "
            "Run [bold]/read <id>[/bold] first (or [bold]/search[/bold] to find one)."
        )
        return
    if not args:
        session.console.print("[yellow]Usage:[/yellow] /discuss <your idea>")
        return
    _run_debate_turn(session, args)


def _run_debate_turn(session: ChatSession, message: str) -> str:
    if session.anchor_paper is None:
        return "Error: no anchor paper loaded. Call load_paper first."
    if not session.idea_seed:
        session.idea_seed = message

    _handle_debate_turn(
        message,
        session.idea_seed,
        session.debate,
        session.memory,
        session.orch,
        session.console,
        paper=session.anchor_paper,
        ideas=session.ideas,
        max_context_tokens=session.max_context_tokens,
    )
    return _debate_round_summary(session.debate)


def _debate_round_summary(debate: DebateHistory) -> str:
    if not debate.rounds:
        return "No debate rounds yet."
    last = debate.rounds[-1]
    if isinstance(last.result, DebateResult):
        return (
            f"Debate round {last.result.round_index} — "
            f"score {last.result.score:.0f}/9. "
            f"Supports: {last.result.supports}. "
            f"Objections: {last.result.objections}."
        )
    if isinstance(last.followup, FollowUpResult):
        parts: list[str] = []
        if last.followup.analyst_conclusion:
            parts.append(f"Analyst: {last.followup.analyst_conclusion}")
        if last.followup.critic_conclusion:
            parts.append(f"Critic: {last.followup.critic_conclusion}")
        return "Follow-up — " + " | ".join(parts)
    return "Debate round recorded."


@register_llm_tool(
    "discuss_idea",
    _function_schema(
        "discuss_idea",
        "Run one Analyst + Critic debate turn about an idea, grounded in the "
        "currently anchored paper. The first call yields a structured debate "
        "(supports, objections, suggestions, score). Subsequent calls produce "
        "prose follow-ups using accumulated context.",
        {
            "idea": {
                "type": "string",
                "description": "User idea or follow-up question about the anchor paper.",
            },
        },
        required=["idea"],
    ),
)
def exec_discuss_idea(session: ChatSession, args: dict[str, Any]) -> str:
    idea = str(args.get("idea", "")).strip()
    if not idea:
        return "Error: idea is required."
    return _run_debate_turn(session, idea)


# ---------------------------------------------------------------- paper


@slash(
    "paper",
    summary="Show the currently anchored paper.",
    usage="/paper",
)
def cmd_paper(session: ChatSession, args: str) -> None:
    if session.anchor_paper is None:
        session.console.print(
            "[dim]No anchor paper loaded yet. Use [/dim][cyan]/read <id>[/cyan][dim].[/dim]"
        )
        return
    render_paper_header(session.console, session.anchor_paper)


# ---------------------------------------------------------------- idea


@slash(
    "idea",
    summary="Manage current debate idea (save / show).",
    usage="/idea save [title] | /idea show",
)
def cmd_idea(session: ChatSession, args: str) -> None:
    parts = args.split(maxsplit=1)
    sub = parts[0] if parts else "show"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "save":
        _save_current_idea(session, rest.strip() or None)
        return
    if sub == "show":
        _show_current_idea(session)
        return
    session.console.print(
        f"[yellow]Unknown:[/yellow] /idea {sub}. Try /idea save or /idea show."
    )


def _save_current_idea(session: ChatSession, title: str | None) -> str:
    if session.anchor_paper is None or not session.debate.rounds:
        session.console.print(
            "[yellow]Need an anchor paper and at least one /discuss round before saving.[/yellow]"
        )
        return "Error: need an anchor paper and at least one debate round."

    final_title = (
        title or session.idea_seed.split("\n", 1)[0][:80] or "Untitled idea"
    )
    idea = session.ideas.create(title=final_title, description=session.idea_seed)
    idea.related_papers = [session.anchor_paper.id]
    session.ideas.save(idea)
    session.current_idea_id = idea.id
    session.memory.idea_id = idea.id
    session.debate.idea_id = idea.id

    opening: DebateResult | None = None
    for rnd in reversed(session.debate.rounds):
        if rnd.result is not None:
            opening = rnd.result
            break
    if opening is not None:
        idea.critic_score = opening.score
        idea.critic_objections = list(opening.objections)
        session.ideas.append_score(
            idea.id,
            opening.score,
            opening.score_reason,
            session_id=session.memory.session_id,
        )
        session.ideas.save(idea)

    session.vectors.upsert(idea)
    session.console.print(
        f"[green]✓[/green] Idea saved: [bold]{idea.title}[/bold] ({idea.id[:8]}…)"
    )
    return f"Saved idea {idea.id[:8]}: {idea.title}"


@register_llm_tool(
    "save_current_idea",
    _function_schema(
        "save_current_idea",
        "Persist the current debate as a saved idea. Requires an anchor paper "
        "and at least one prior discuss_idea round.",
        {
            "title": {
                "type": "string",
                "description": "Optional title; if omitted, derived from the seed idea.",
            },
        },
    ),
)
def exec_save_current_idea(session: ChatSession, args: dict[str, Any]) -> str:
    title = args.get("title")
    if title is not None:
        title = str(title).strip() or None
    return _save_current_idea(session, title)


@register_llm_tool(
    "list_ideas",
    _function_schema(
        "list_ideas",
        "List all saved ideas grouped by status.",
        {},
    ),
)
def exec_list_ideas(session: ChatSession, args: dict[str, Any]) -> str:
    ideas = session.ideas.list_all()
    if not ideas:
        return "No ideas saved yet."
    lines = ["Saved ideas:"]
    for idea in ideas:
        score = (
            f"{idea.critic_score:.0f}/9" if idea.critic_score is not None else "—"
        )
        lines.append(
            f"- {idea.id[:8]} [{idea.status}] {idea.title} (score {score})"
        )
    # also render in the console for the user
    run_ideas_list(session.cfg, session.console)
    return "\n".join(lines)


def _show_current_idea(session: ChatSession) -> None:
    if session.current_idea_id is None:
        if session.idea_seed:
            session.console.print(
                f"[dim]Unsaved debate idea:[/dim] {session.idea_seed[:120]}\n"
                "[dim]Use[/dim] /idea save [title] [dim]to persist it.[/dim]"
            )
        else:
            session.console.print(
                "[dim]No active idea. Use[/dim] /discuss <idea> [dim]to start one.[/dim]"
            )
        return
    run_ideas_show(session.cfg, session.console, session.current_idea_id)


# ---------------------------------------------------------------- ideas (library)


@slash(
    "ideas",
    summary="Browse the saved-ideas library.",
    usage="/ideas [list] | /ideas show <id> | /ideas update <id> --status <status>",
)
def cmd_ideas(session: ChatSession, args: str) -> None:
    parts = args.split()
    if not parts or parts[0] == "list":
        run_ideas_list(session.cfg, session.console)
        return
    sub = parts[0]
    if sub == "show":
        if len(parts) < 2:
            session.console.print("[yellow]Usage:[/yellow] /ideas show <id>")
            return
        run_ideas_show(session.cfg, session.console, parts[1])
        return
    if sub == "update":
        _ideas_update(session, parts[1:])
        return
    session.console.print(
        f"[yellow]Unknown:[/yellow] /ideas {sub}. Try /ideas list|show|update."
    )


def _ideas_update(session: ChatSession, tokens: list[str]) -> None:
    if not tokens:
        session.console.print(
            "[yellow]Usage:[/yellow] /ideas update <id> --status <status> | --feedback <note>"
        )
        return
    idea_id = tokens[0]
    status: IdeaStatus | None = None
    feedback: str | None = None
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--status" and i + 1 < len(tokens):
            candidate = tokens[i + 1]
            if candidate not in IDEA_STATUSES:
                session.console.print(
                    f"[red]Error:[/red] Invalid status. Choose: {', '.join(IDEA_STATUSES)}"
                )
                return
            status = candidate
            i += 2
        elif token == "--feedback" and i + 1 < len(tokens):
            feedback = " ".join(tokens[i + 1 :])
            break
        else:
            i += 1
    if status is None and feedback is None:
        session.console.print(
            "[yellow]Nothing to update. Use --status or --feedback.[/yellow]"
        )
        return
    run_ideas_update(
        session.cfg,
        session.console,
        idea_id,
        status=status,
        feedback=feedback,
    )


# ---------------------------------------------------------------- help


@slash(
    "help",
    summary="Show this command list.",
    usage="/help",
)
def cmd_help(session: ChatSession, args: str) -> None:
    table = Table(title="Slash commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("What it does")
    table.add_column("Usage", style="dim")
    for name in sorted(SLASH_COMMANDS):
        cmd = SLASH_COMMANDS[name]
        table.add_row(f"/{name}", cmd.summary, cmd.usage)
    session.console.print(table)
    session.console.print(
        "\n[dim]Plain text (no leading /) is sent to the LLM with the current "
        "conversation as context.[/dim]"
    )


# ---------------------------------------------------------------- helpers


def _analysis_to_dict(result: AnalysisResult) -> dict[str, Any]:
    data: dict[str, Any] = asdict(result)
    data["claimed_vs_evidence"] = [
        {"claim": p.claim, "evidence": p.evidence} for p in result.claimed_vs_evidence
    ]
    return data


def _critique_to_dict(result: CritiqueResult) -> dict[str, Any]:
    return asdict(result)
