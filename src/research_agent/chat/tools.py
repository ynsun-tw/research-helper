"""Tool registry: slash command handlers + LLM-callable executors.

Each tool has a slash interface (``/name args`` for explicit invocation) and
optionally an LLM tool-calling interface (JSON-Schema parameters + executor
that returns a text result for the LLM's next reasoning step).

``SLASH_COMMANDS`` is consumed by router.py for slash dispatch and ``/help``.
``LLM_TOOLS`` exposes OpenAI-compatible function schemas to the agent loop.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.table import Table

from research_agent.agents.analyst import AnalysisResult
from research_agent.agents.critic import CritiqueResult
from research_agent.agents.debate import DebateHistory, DebateResult, FollowUpResult
from research_agent.agents.illustrator import FigureDraft
from research_agent.agents.meta_memory import MetaMemory
from research_agent.agents.scribe import Draft
from research_agent.agents.searcher import SearchSuggestion
from research_agent.chat.session import ChatSession
from research_agent.cli_ideas import run_ideas_list, run_ideas_show, run_ideas_update
from research_agent.cli_services import _handle_debate_turn, _load_anchor_paper
from research_agent.core.idea import IDEA_STATUSES, IdeaStatus
from research_agent.core.llm import LLMError
from research_agent.core.loader import PaperLoadError
from research_agent.core.paper import Paper
from research_agent.core.paper_resolver import parse_search_mode, search_arxiv_papers
from research_agent.search.arxiv_search import ArxivSearchError, ArxivSearchHit
from research_agent.search.semantic_scholar import SemanticScholarSearcher
from research_agent.storage.discussions import DiscussionMessage
from research_agent.storage.reading_queue import ALLOWED_STATUSES, QueueEntry, QueueStatus
from research_agent.storage.searches import StoredSearchQuery
from research_agent.ui.formatting import render_paper_header, render_read_report

SlashHandler = Callable[[ChatSession, str], None]
LLMExecutor = Callable[[ChatSession, dict[str, Any]], str]


@dataclass(frozen=True)
class SlashCommand:
    name: str
    handler: SlashHandler
    summary: str
    usage: str


@dataclass(frozen=True)
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
    summary=(
        "Search arXiv + LLM relevance score (logs to history; flags already-read). "
        "Add --mode theoretical|applied|group:<author> to bias the candidate set."
    ),
    usage="/search [--mode theoretical|applied|group:<author>] <keywords>",
)
def cmd_search(session: ChatSession, args: str) -> None:
    if not args:
        session.console.print(
            "[yellow]Usage:[/yellow] /search [--mode <m>] <keywords>"
        )
        return
    mode, query = _extract_mode_flag(args)
    if not query:
        session.console.print("[yellow]Usage:[/yellow] /search <keywords>")
        return
    if mode is not None:
        parsed = parse_search_mode(mode)
        if parsed.warning:
            session.console.print(f"[yellow]{parsed.warning}[/yellow]")
        elif parsed.kind == "group":
            session.console.print(
                f"[dim]Biasing search toward author {parsed.author!r}.[/dim]"
            )
        elif parsed.kind is not None:
            session.console.print(
                f"[dim]Biasing search toward {parsed.kind} papers.[/dim]"
            )
    session.last_search_query = query
    try:
        with session.console.status("[bold]Searching arXiv…[/bold]"):
            hits = search_arxiv_papers(query, max_results=5, mode=mode)
    except (PaperLoadError, ArxivSearchError) as exc:
        session.console.print(f"[red]Error:[/red] {exc}")
        return
    if not hits:
        session.console.print(f"[yellow]No results for[/yellow] {query!r}.")
        session.searches.record(
            query, hits, source="arxiv", session_id=session.memory.session_id
        )
        session.memory.append("system", _search_summary(query, hits))
        return

    scored = _score_hits(session, query, hits)
    session.searches.record(
        query, scored, source="arxiv", session_id=session.memory.session_id
    )
    sorted_hits = _sort_by_score(scored)
    already_read = session.searches.already_read([h.arxiv_id for h in sorted_hits])
    _render_search_hits(session, query, sorted_hits, already_read=already_read)
    _surface_activation_alerts(session, sorted_hits)
    session.memory.append(
        "system",
        _search_summary(query, sorted_hits, already_read=already_read),
    )


def _extract_mode_flag(args: str) -> tuple[str | None, str]:
    """Parse ``--mode <value>`` out of a slash-command argument string.

    Returns ``(mode, remaining_query)``. The mode can appear anywhere
    in ``args``; supports ``--mode foo``, ``--mode=foo``, and quoted
    multi-word values (``--mode "group:Andrej Karpathy"``). Uses
    :func:`shlex.split` so quoting works the same way as a shell.
    Falls back to whitespace split if the input has unbalanced quotes
    (so the parser never aborts the user's search).
    """
    try:
        tokens = shlex.split(args, posix=True)
    except ValueError:
        tokens = args.split()
    mode: str | None = None
    rest: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--mode" and i + 1 < len(tokens):
            mode = tokens[i + 1]
            i += 2
            continue
        if tok.startswith("--mode="):
            mode = tok[len("--mode="):]
            i += 1
            continue
        rest.append(tok)
        i += 1
    return mode, " ".join(rest).strip()


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
        "candidate papers (id, title, year) but does not load them. Optional "
        "mode biases the candidate set: 'theoretical' (analysis / proofs), "
        "'applied' (benchmarks / experiments), 'group:<author-name>' "
        "(papers by a specific author).",
        {
            "query": {"type": "string", "description": "Keywords or title fragments."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return (default 5, max 10).",
                "minimum": 1,
                "maximum": 10,
            },
            "mode": {
                "type": "string",
                "description": (
                    "Optional search bias. One of: 'theoretical', 'applied', "
                    "'group:<author-name>'. Unknown values are ignored."
                ),
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
    mode_raw = args.get("mode")
    mode = str(mode_raw).strip() if mode_raw else None
    if mode:
        parsed = parse_search_mode(mode)
        if parsed.warning:
            return f"Error: {parsed.warning}"
    session.last_search_query = query
    try:
        with session.console.status(f"[bold]Searching arXiv:[/bold] {query}…"):
            hits = search_arxiv_papers(query, max_results=max_results, mode=mode)
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
    _surface_activation_alerts(session, sorted_hits)
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


# --------------------------------------------------------- dynamic refinement


def _build_refinement_context(session: ChatSession) -> str:
    """Pull a transcript snippet from working memory for the Searcher.

    Returns ``""`` when there's nothing usable (slash refuses to make
    a no-context refinement). Delegates to the Orchestrator helper
    which already knows how to crop messages / chars.
    """
    return session.orch.extract_search_context(session.memory)


def _format_suggestion(s: SearchSuggestion) -> str:
    """Compact one-block pretty-print for the suggestion banner."""
    mode_part = f" [dim](mode: {s.mode})[/dim]" if s.mode else ""
    conf = f"{s.confidence:.0%}" if s.confidence else "—"
    return (
        f"[bold]Suggested next search:[/bold] [cyan]{s.query}[/cyan]"
        f"{mode_part}\n"
        f"[dim]Reason:[/dim] {s.reason or '(no reason given)'} "
        f"[dim](confidence {conf})[/dim]"
    )


def _format_search_args(query: str, mode: str | None) -> str:
    """Compose the argv string that would feed `cmd_search`."""
    if not mode:
        return query
    # Quote group:<author> to survive shlex when the author has spaces.
    if " " in mode:
        return f'--mode "{mode}" {query}'
    return f"--mode {mode} {query}"


@slash(
    "refine",
    summary="Ask Searcher to suggest the next search query from recent discussion.",
    usage="/refine",
)
def cmd_refine(session: ChatSession, args: str) -> None:
    context = _build_refinement_context(session)
    if not context:
        session.console.print(
            "[dim]Not enough conversation yet. Discuss or read a paper first, "
            "then run /refine.[/dim]"
        )
        return

    try:
        with session.console.status(
            "[bold]Searcher: building refined query…[/bold]"
        ):
            suggestion = session.searcher.suggest_refinement(
                context, previous_query=session.last_search_query or None
            )
    except LLMError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")
        return

    if not suggestion.query:
        session.console.print(
            "[yellow]Searcher returned no refined query.[/yellow] "
            "[dim]Try discussing more, or run /search manually.[/dim]"
        )
        return

    session.console.print(_format_suggestion(suggestion))
    session.memory.append(
        "system",
        f"Refinement suggestion: query={suggestion.query!r} "
        f"mode={suggestion.mode} reason={suggestion.reason!r}",
    )
    # Interactive accept/edit/skip - the input loop is intentionally simple
    # so non-interactive callers (tests / LLM tool) can short-circuit.
    try:
        choice = session.input_fn(
            "Run this search? [y]es / [e]dit / [s]kip > "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "s"
    if choice in ("", "s", "skip", "n", "no"):
        session.console.print("[dim]Skipped.[/dim]")
        return
    if choice in ("y", "yes", ""):
        cmd_search(session, _format_search_args(suggestion.query, suggestion.mode))
        return
    if choice.startswith("e"):
        edited = session.input_fn(
            f"Edit (current: {suggestion.query}) > "
        ).strip()
        if not edited:
            session.console.print("[dim]No edit; skipping.[/dim]")
            return
        cmd_search(session, _format_search_args(edited, suggestion.mode))
        return
    session.console.print("[dim]Unrecognised choice; skipping.[/dim]")


@register_llm_tool(
    "suggest_search_refinement",
    _function_schema(
        "suggest_search_refinement",
        "Read the recent discussion transcript and propose the next search "
        "query (with optional bias mode and a one-sentence reason). Returns "
        "JSON with fields: query, mode, reason, confidence. The tool does "
        "NOT execute the search - chain into search_arxiv afterwards if the "
        "user accepts.",
        {},
    ),
)
def exec_suggest_search_refinement(
    session: ChatSession, args: dict[str, Any]
) -> str:
    context = _build_refinement_context(session)
    if not context:
        return "Error: not enough discussion context to refine. Discuss or load a paper first."
    try:
        suggestion = session.searcher.suggest_refinement(
            context, previous_query=session.last_search_query or None
        )
    except LLMError as exc:
        return f"Error: {exc}"
    if not suggestion.query:
        return "Searcher returned no refinement (insufficient signal in discussion)."
    # Render to the console so the user sees the model's reasoning.
    session.console.print(_format_suggestion(suggestion))
    return (
        "Refinement suggestion: "
        f"query={suggestion.query!r}, mode={suggestion.mode}, "
        f"confidence={suggestion.confidence:.2f}, reason={suggestion.reason!r}. "
        "Call search_arxiv with this query (and mode if set) to run it."
    )


# --------------------------------------------------------- insights


def _parse_since_days(raw: str) -> int | None:
    """Parse ``--since 30d|7d|6m|1y|all`` to a day count (or None for all)."""
    s = raw.strip().lower()
    if not s or s in ("all", "alltime", "all-time"):
        return None
    unit_map = {"d": 1, "w": 7, "m": 30, "y": 365}
    if s[-1] in unit_map and s[:-1].isdigit():
        return int(s[:-1]) * unit_map[s[-1]]
    if s.isdigit():
        return int(s)
    raise ValueError(f"Unrecognised --since value: {raw!r}")


def _build_insights(session: ChatSession, since_days: int | None) -> str:
    """Run MetaMemory and return its Markdown report."""
    meta = MetaMemory(session.db)
    report = meta.compute(since_days=since_days)
    return report.to_markdown()


@slash(
    "insights",
    summary="Markdown research summary: papers, ideas, discussion activity.",
    usage="/insights [--since 30d|7d|6m|all]",
)
def cmd_insights(session: ChatSession, args: str) -> None:
    since_days: int | None = None
    tokens = args.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--since" and i + 1 < len(tokens):
            try:
                since_days = _parse_since_days(tokens[i + 1])
            except ValueError as exc:
                session.console.print(f"[red]Error:[/red] {exc}")
                return
            i += 2
            continue
        if tok.startswith("--since="):
            try:
                since_days = _parse_since_days(tok[len("--since="):])
            except ValueError as exc:
                session.console.print(f"[red]Error:[/red] {exc}")
                return
            i += 1
            continue
        session.console.print(
            f"[yellow]Unknown flag:[/yellow] {tok}. "
            "Usage: /insights [--since 30d|7d|6m|all]"
        )
        return
    markdown = _build_insights(session, since_days)
    # Render Markdown for the user; the LLM (via system memory) gets the
    # raw text below so it can answer follow-up questions about the report.
    try:
        from rich.markdown import Markdown
        session.console.print(Markdown(markdown))
    except ImportError:
        session.console.print(markdown)
    session.memory.append("system", f"Insights report (period={since_days}):\n{markdown}")


@register_llm_tool(
    "research_insights",
    _function_schema(
        "research_insights",
        "Compute a research-activity summary (papers, ideas, discussion "
        "stats) from local storage and return it as Markdown. Use when the "
        "user asks 'how am I doing', 'what have I been reading', or 'show "
        "me my recent activity'. Optional `since_days` filters to recent "
        "activity (e.g. 30 for last month).",
        {
            "since_days": {
                "type": "integer",
                "description": "Optional day window. Omit for all-time.",
                "minimum": 1,
                "maximum": 3650,
            },
        },
    ),
)
def exec_research_insights(session: ChatSession, args: dict[str, Any]) -> str:
    since_raw = args.get("since_days")
    since_days: int | None = None
    if since_raw is not None:
        try:
            since_days = max(1, min(int(since_raw), 3650))
        except (TypeError, ValueError):
            return "Error: since_days must be an integer (days)."
    markdown = _build_insights(session, since_days)
    # Render to the console too so the user sees what the model is reading.
    try:
        from rich.markdown import Markdown
        session.console.print(Markdown(markdown))
    except ImportError:
        session.console.print(markdown)
    return markdown


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


def _surface_parked_idea_alerts(session: ChatSession, paper: Paper) -> None:
    """Print a one-line banner if the loaded paper looks related to any
    shelved/waiting idea (M3 T3.4.1.2).

    The banner is intentionally compact (one header line + one bullet
    per alert, capped at 3) so it doesn't dominate the /read report.
    Failures are swallowed (best-effort): if the vector store is
    misconfigured or empty, the user shouldn't see an error mid-read.
    """
    try:
        context = f"{paper.title}\n{paper.abstract}".strip()
        if not context:
            return
        threshold = _alert_threshold(session)
        associations = session.keeper.check_associations(
            context, threshold=threshold, limit=3
        )
        if not associations:
            return
        banner = session.keeper.format_associations(associations)
        if not banner:
            return
        session.console.print(banner)
        # Drop a system message into working memory so the LLM can also
        # see the alert if the user keeps chatting after /read.
        title_list = ", ".join(a.idea.title for a in associations)
        session.memory.append(
            "system",
            f"Related parked ideas surfaced for {paper.id}: {title_list}",
        )
    except Exception:
        # Alerts are best-effort - any failure must NOT abort the read.
        return


def _surface_activation_alerts(
    session: ChatSession, hits: list[ArxivSearchHit]
) -> None:
    """Print a one-line banner if any incoming search hit looks like it
    satisfies an `activation_conditions` phrase on a shelved/waiting idea
    (M3 T3.4.2.3).

    The match is intentionally simple - case-insensitive substring of the
    user-supplied condition phrase against ``hit.title + ' ' + hit.abstract``.
    Free-form phrases like "FineWeb-Edu dataset" or "1B model checkpoint"
    almost always survive verbatim in the new paper if it actually delivers
    them, and we have no LLM-budget to spend on this pre-filter step.

    Caps the banner at 3 lines (most-recent hits first) so it never
    drowns out the search table. All failures swallowed.
    """
    try:
        if not hits:
            return
        ideas = [
            i
            for i in session.ideas.list_all()
            if i.activation_conditions and i.status in ("shelved", "waiting")
        ]
        if not ideas:
            return
        matches: list[tuple[str, str, ArxivSearchHit, str]] = []
        for hit in hits:
            haystack = f"{hit.title}\n{hit.abstract or ''}".lower()
            if not haystack.strip():
                continue
            for idea in ideas:
                for condition in idea.activation_conditions:
                    needle = condition.strip().lower()
                    if needle and needle in haystack:
                        matches.append((idea.id, idea.title, hit, condition))
                        # Only first matching condition per (idea, hit)
                        break
        if not matches:
            return
        session.console.print(
            "[bold yellow]Shelved idea(s) may have an unblock:[/bold yellow]"
        )
        for idea_id, idea_title, hit, condition in matches[:3]:
            session.console.print(
                f"  - [cyan]{hit.arxiv_id}[/cyan] [italic]{hit.title[:60]}[/italic] "
                f"matches condition “{condition}” on "
                f"[bold]{idea_title}[/bold] — "
                f"/idea show {idea_id[:8]}"
            )
        # Memory note so the LLM can pick up the thread if the user keeps
        # chatting after /search.
        titles = ", ".join({m[1] for m in matches[:3]})
        session.memory.append(
            "system",
            f"Search hits may satisfy activation conditions on: {titles}",
        )
    except Exception:
        return


def _alert_threshold(session: ChatSession) -> float:
    """Resolve the association similarity threshold from Config.

    Config carries the source-of-truth value (settable via
    ``research config set alert_threshold 0.85``); the field is
    pydantic-clamped to [0,1] on load so callers don't need to
    re-validate. Falls back to the MemoryKeeper spec default if a
    test injects a stub Config without the field.
    """
    from research_agent.agents.memory_keeper import DEFAULT_ASSOCIATION_THRESHOLD

    cfg_value = getattr(session.cfg, "alert_threshold", None)
    if isinstance(cfg_value, (int, float)):
        return max(0.0, min(1.0, float(cfg_value)))
    return DEFAULT_ASSOCIATION_THRESHOLD


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
    _surface_parked_idea_alerts(session, paper)
    session.memory.append(
        "system",
        f"Loaded and analyzed paper: {paper.title} ({paper.id})",
    )
    # Paper ids are prefixed (arxiv:1706.03762 or local:<sha1>) but the
    # reading queue stores raw arXiv ids. Strip the prefix before lookup.
    queue_lookup_id = (
        paper.id[len("arxiv:"):] if paper.id.startswith("arxiv:") else paper.id
    )
    if session.queue.get(queue_lookup_id) is not None:
        session.queue.set_status(queue_lookup_id, "done")
        session.console.print(
            f"[dim]✓ Marked[/dim] {queue_lookup_id} "
            "[dim]as done in the reading queue.[/dim]"
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
    usage=(
        "/ideas [list] | /ideas show <id> | /ideas update <id> "
        "[--status <s>] [--feedback <note>] [--condition <phrase>] "
        "[--clear-conditions]"
    ),
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
            "[yellow]Usage:[/yellow] /ideas update <id> "
            "[--status <status>] [--feedback <note>] "
            '[--condition "<phrase>"] [--clear-conditions]'
        )
        return
    idea_id = tokens[0]
    status: IdeaStatus | None = None
    feedback: str | None = None
    conditions: list[str] = []
    clear_conditions = False
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
        elif token == "--condition" and i + 1 < len(tokens):
            # Greedy: consume everything until the next --flag so phrases can
            # contain spaces without quoting (e.g. `--condition needs FineWeb
            # dataset`). Stops at the next `--xxx` flag to allow chaining
            # multiple conditions in one command.
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                j += 1
            conditions.append(" ".join(tokens[i + 1 : j]))
            i = j
        elif token == "--clear-conditions":
            clear_conditions = True
            i += 1
        elif token == "--feedback" and i + 1 < len(tokens):
            feedback = " ".join(tokens[i + 1 :])
            break
        else:
            i += 1
    if (
        status is None
        and feedback is None
        and not conditions
        and not clear_conditions
    ):
        session.console.print(
            "[yellow]Nothing to update. Use --status, --feedback, --condition, "
            "or --clear-conditions.[/yellow]"
        )
        return
    run_ideas_update(
        session.cfg,
        session.console,
        idea_id,
        status=status,
        feedback=feedback,
        conditions=conditions or None,
        clear_conditions=clear_conditions,
    )


# ---------------------------------------------------------- writing pipeline
#
# Phase 2 tools (draft / figure / check / revise / save) push every
# CLI writing operation into the agent loop. Key design rules:
#
# * Heavy LLM calls (draft / figure / revise) cache their bouquet in
#   ``session.recent_drafts`` / ``recent_figures`` / ``recent_revisions``.
#   This lets follow-up tools refer to "latest" without the user
#   remembering version ids.
# * The filesystem is touched in exactly one place: ``save_draft_to_file``.
#   No tool here silently writes drafts to disk.
# * ``latest`` / ``latest:<section>`` / ``latest:<section>:<version>``
#   is the canonical reference syntax shared by ``draft_section``'s
#   ``check_against``, ``check_self_plagiarism``'s ``target``, and
#   ``revise_draft``'s ``target``. Resolution lives in
#   ``_resolve_latest_text`` so the rules stay in one place.


def _truncate(body: str, *, limit: int = 400) -> str:
    body = (body or "").strip()
    if len(body) <= limit:
        return body
    return body[: limit - 1].rstrip() + "…"


def _parse_latest_ref(ref: str) -> tuple[str | None, str | None]:
    """Parse ``latest`` / ``latest:<section>`` / ``latest:<section>:<version>``.

    Returns ``(section, version)`` with ``None`` meaning "any / first".
    Non-``latest`` strings yield ``(None, None)`` so callers can fall
    back to path-based resolution.
    """
    parts = ref.split(":")
    if not parts or parts[0].strip().lower() != "latest":
        return (None, None)
    section = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    version = parts[2].strip().upper() if len(parts) > 2 and parts[2].strip() else None
    return (section, version)


def _resolve_latest_draft(
    session: ChatSession, ref: str
) -> tuple[str, str, str] | None:
    """Pull text for a ``latest[:section[:version]]`` reference.

    Returns ``(section, version, text)`` or ``None`` if the cache
    cannot satisfy the ref. Resolution rules:

    * ``latest`` alone → first cached section, first version.
    * ``latest:<section>`` → that section, first version.
    * ``latest:<section>:<version>`` → exact match.
    """
    from research_agent.agents.scribe import normalize_section

    section_ref, version_ref = _parse_latest_ref(ref)
    if section_ref is None and version_ref is None and not ref.lower().startswith(
        "latest"
    ):
        return None
    if not session.recent_drafts:
        return None
    if section_ref is None:
        section_key = next(iter(session.recent_drafts))
    else:
        try:
            section_key = normalize_section(section_ref)
        except ValueError:
            return None
        if section_key not in session.recent_drafts:
            return None
    bouquet = session.recent_drafts[section_key]
    if not bouquet:
        return None
    if version_ref is None:
        chosen = bouquet[0]
    else:
        match = [d for d in bouquet if d.version.upper() == version_ref]
        if not match:
            return None
        chosen = match[0]
    return (section_key, chosen.version, chosen.text)


def _resolve_check_targets(
    session: ChatSession, raws: list[str], tmpdir: Path
) -> tuple[list[Path], list[str]]:
    """Resolve ``check_against`` items to Paths and return (paths, errors)."""
    paths: list[Path] = []
    errors: list[str] = []
    for raw in raws:
        if not raw or not raw.strip():
            continue
        raw = raw.strip()
        if raw.lower().startswith("latest"):
            resolved = _resolve_latest_draft(session, raw)
            if resolved is None:
                errors.append(
                    f"could not resolve '{raw}' — no cached drafts (call "
                    f"draft_section first)"
                )
                continue
            section, version, text = resolved
            tmp_path = tmpdir / f"latest_{section}_{version}.md"
            tmp_path.write_text(text, encoding="utf-8")
            paths.append(tmp_path)
        else:
            p = Path(raw).expanduser()
            if not p.exists():
                errors.append(f"file not found: {p}")
                continue
            paths.append(p)
    return paths, errors


def _format_draft_summary(section: str, drafts: list[Draft]) -> str:
    """One-paragraph summary the LLM can read after a draft tool call."""
    if not drafts:
        return f"Scribe produced no drafts for {section}."
    lines = [f"Drafted {len(drafts)} variant(s) of `{section}`:"]
    for d in drafts:
        excerpt = _truncate(d.text, limit=240)
        lines.append(
            f"- Version {d.version} ({d.variant_label}, ~{d.word_count} words): "
            f"{excerpt}"
        )
    lines.append(
        "Cached in session. Ask to save a specific version with "
        "save_draft_to_file (or run check_self_plagiarism / revise_draft "
        "against 'latest:' refs)."
    )
    return "\n".join(lines)


def _format_figure_summary(figure_type: str, drafts: list[FigureDraft]) -> str:
    if not drafts:
        return f"Illustrator produced no drafts for {figure_type}."
    lines = [f"Drafted {len(drafts)} variant(s) of `{figure_type}` figure:"]
    for d in drafts:
        head = (
            f"- Version {d.version} ({d.style_label or 'no label'}, "
            f"{d.code_language})"
        )
        if d.target_model:
            head += f", target={d.target_model}"
        lines.append(head)
        if d.notes:
            lines.append(f"  notes: {_truncate(d.notes, limit=140)}")
        lines.append(f"  code size: {len(d.code)} chars")
    lines.append(
        "Cached in session. Use save_draft_to_file(kind='figure', "
        "figure_type=…) to write them out."
    )
    return "\n".join(lines)


@register_llm_tool(
    "draft_section",
    _function_schema(
        "draft_section",
        "Draft a paper section in the user's voice (N variants, cached, "
        "not saved to disk).",
        {
            "section": {
                "type": "string",
                "description": (
                    "abstract|introduction|related_work|method|results|"
                    "discussion|conclusion (aliases 'intro' / 'methods' / "
                    "'experiments' accepted)."
                ),
            },
            "context": {
                "type": "string",
                "description": "Research context to ground the draft in.",
            },
            "target_words": {
                "type": "integer",
                "description": "Target words per draft (±20%). Default 300.",
                "minimum": 50,
                "maximum": 2000,
            },
            "versions": {
                "type": "integer",
                "description": "Variants (1-5). Default 3.",
                "minimum": 1,
                "maximum": 5,
            },
            "check_against": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Paths or `latest` refs ('latest', 'latest:<section>', "
                    "'latest:<section>:<version>') for consistency check."
                ),
            },
        },
        required=["section"],
    ),
)
def exec_draft_section(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.agents.scribe import normalize_section
    from research_agent.cli_write import run_write

    raw_section = str(args.get("section") or "").strip()
    if not raw_section:
        return "Error: section is required."
    try:
        section = normalize_section(raw_section)
    except ValueError as exc:
        return f"Error: {exc}"

    context = str(args.get("context") or "")
    target_words = int(args.get("target_words") or 300)
    versions = int(args.get("versions") or 3)
    if not 1 <= versions <= 5:
        return "Error: versions must be between 1 and 5."
    check_against_raw = args.get("check_against") or []
    if not isinstance(check_against_raw, list):
        return "Error: check_against must be an array of strings."
    check_against_strs = [str(x) for x in check_against_raw]

    tmpdir = Path(tempfile.mkdtemp(prefix="rabot-draft-"))
    try:
        paths, errors = _resolve_check_targets(
            session, check_against_strs, tmpdir
        )
        if errors:
            return "Error resolving check_against:\n" + "\n".join(
                f"- {e}" for e in errors
            )
        try:
            result = run_write(
                session.cfg,
                session.console,
                section=section,
                context=context,
                check_against=paths or None,
                target_words=target_words,
                versions=versions,
                output=None,
                parallel=True,
            )
        except LLMError as exc:
            return f"Error from Scribe: {exc}"

        session.recent_drafts[result.section] = list(result.drafts)
        return _format_draft_summary(result.section, result.drafts)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@register_llm_tool(
    "draft_figure",
    _function_schema(
        "draft_figure",
        "Generate figure code (TikZ for architecture, matplotlib for "
        "result, text-to-image prompt for concept). N variants, cached.",
        {
            "figure_type": {
                "type": "string",
                "description": (
                    "architecture|result|concept (aliases 'pipeline', "
                    "'plot', 'schematic' accepted)."
                ),
            },
            "description": {
                "type": "string",
                "description": "What to draw.",
            },
            "data": {
                "type": "string",
                "description": "Quantitative payload for result figures.",
            },
            "versions": {
                "type": "integer",
                "description": "Variants (1-4). Default 2.",
                "minimum": 1,
                "maximum": 4,
            },
            "verify": {
                "type": "boolean",
                "description": (
                    "If true and figure_type=result, actually run each "
                    "draft (30s, Agg backend) and report pass/fail."
                ),
            },
        },
        required=["figure_type", "description"],
    ),
)
def exec_draft_figure(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.agents.illustrator import normalize_figure_type
    from research_agent.cli_figure import run_figure

    raw_type = str(args.get("figure_type") or "").strip()
    description = str(args.get("description") or "").strip()
    if not raw_type:
        return "Error: figure_type is required."
    if not description:
        return "Error: description is required."
    try:
        figure_type = normalize_figure_type(raw_type)
    except ValueError as exc:
        return f"Error: {exc}"
    data = str(args.get("data") or "")
    versions = int(args.get("versions") or 2)
    if not 1 <= versions <= 4:
        return "Error: versions must be between 1 and 4."
    verify = bool(args.get("verify") or False)

    try:
        result = run_figure(
            session.cfg,
            session.console,
            figure_type=figure_type,
            description=description,
            data=data,
            versions=versions,
            output=None,
            verify=verify,
            parallel=True,
        )
    except LLMError as exc:
        return f"Error from Illustrator: {exc}"

    session.recent_figures[result.figure_type] = list(result.drafts)
    summary = _format_figure_summary(result.figure_type, result.drafts)
    if result.verifications:
        ok = sum(1 for v in result.verifications if v.ok)
        fail = len(result.verifications) - ok
        summary += f"\nVerification: {ok} passed, {fail} failed."
    return summary


@register_llm_tool(
    "save_draft_to_file",
    _function_schema(
        "save_draft_to_file",
        "Write a cached draft / figure / revision to a Markdown file. "
        "Only call after the user explicitly asks to save.",
        {
            "path": {
                "type": "string",
                "description": "Destination (~ expanded, parents created).",
            },
            "kind": {
                "type": "string",
                "enum": ["section", "figure", "revision"],
                "description": (
                    "section|figure|revision. Omit to infer when "
                    "unambiguous."
                ),
            },
            "section": {
                "type": "string",
                "description": (
                    "Section name. Required if multiple sections cached."
                ),
            },
            "figure_type": {
                "type": "string",
                "description": (
                    "Figure type. Required if multiple figures cached."
                ),
            },
            "version": {
                "type": "string",
                "description": (
                    "Variant letter (A/B/C). Omit to save the whole "
                    "bouquet. Ignored for revision."
                ),
            },
        },
        required=["path"],
    ),
)
def exec_save_draft_to_file(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.agents.illustrator import normalize_figure_type
    from research_agent.agents.scribe import normalize_section
    from research_agent.cli_figure import _persist_drafts as persist_figures
    from research_agent.cli_write import _persist_drafts as persist_sections

    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return "Error: path is required."
    out_path = Path(raw_path).expanduser()
    kind = str(args.get("kind") or "").strip().lower()

    if not kind:
        n_sections = len(session.recent_drafts)
        n_figures = len(session.recent_figures)
        n_revisions = len(session.recent_revisions)
        present = [k for k, n in (
            ("section", n_sections),
            ("figure", n_figures),
            ("revision", n_revisions),
        ) if n]
        if not present:
            return (
                "Error: nothing in the session cache to save. Run "
                "draft_section / draft_figure / revise_draft first."
            )
        if len(present) > 1:
            return (
                "Error: cache holds " + ", ".join(present)
                + ". Please specify kind (section|figure|revision)."
            )
        kind = present[0]

    if kind == "section":
        return _save_section(session, args, out_path, normalize_section,
                             persist_sections)
    if kind == "figure":
        return _save_figure(session, args, out_path, normalize_figure_type,
                            persist_figures)
    if kind == "revision":
        return _save_revision(session, args, out_path, normalize_section)
    return f"Error: unknown kind {kind!r}. Use section, figure, or revision."


def _save_section(
    session: ChatSession,
    args: dict[str, Any],
    out_path: Path,
    normalize: Callable[[str], str],
    persist: Callable[[Path, str, list[Draft]], None],
) -> str:
    if not session.recent_drafts:
        return "Error: no section drafts in cache. Call draft_section first."
    section_raw = str(args.get("section") or "").strip()
    if section_raw:
        try:
            section = normalize(section_raw)
        except ValueError as exc:
            return f"Error: {exc}"
        if section not in session.recent_drafts:
            return (
                f"Error: no cached draft for section {section!r}. "
                f"Available: {', '.join(sorted(session.recent_drafts))}."
            )
    elif len(session.recent_drafts) == 1:
        section = next(iter(session.recent_drafts))
    else:
        return (
            "Error: multiple sections cached "
            f"({', '.join(sorted(session.recent_drafts))}); specify section."
        )
    bouquet = session.recent_drafts[section]
    version = str(args.get("version") or "").strip().upper()
    if version:
        match = [d for d in bouquet if d.version.upper() == version]
        if not match:
            available = ", ".join(d.version for d in bouquet)
            return (
                f"Error: no version {version!r} for {section!r}. "
                f"Available: {available}."
            )
        chosen = [match[0]]
    else:
        chosen = list(bouquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    persist(out_path, section, chosen)
    session.console.print(
        f"[green]✓[/green] Wrote {len(chosen)} draft(s) to [bold]{out_path}[/bold]"
    )
    return (
        f"Saved {len(chosen)} {section} draft(s)"
        + (f" (version {version})" if version else "")
        + f" to {out_path}."
    )


def _save_figure(
    session: ChatSession,
    args: dict[str, Any],
    out_path: Path,
    normalize: Callable[[str], str],
    persist: Callable[[Path, str, str, list[FigureDraft], list[Any]], None],
) -> str:
    if not session.recent_figures:
        return "Error: no figure drafts in cache. Call draft_figure first."
    type_raw = str(args.get("figure_type") or "").strip()
    if type_raw:
        try:
            figure_type = normalize(type_raw)
        except ValueError as exc:
            return f"Error: {exc}"
        if figure_type not in session.recent_figures:
            return (
                f"Error: no cached figure for type {figure_type!r}. "
                f"Available: {', '.join(sorted(session.recent_figures))}."
            )
    elif len(session.recent_figures) == 1:
        figure_type = next(iter(session.recent_figures))
    else:
        return (
            "Error: multiple figure types cached "
            f"({', '.join(sorted(session.recent_figures))}); specify "
            "figure_type."
        )
    bouquet = session.recent_figures[figure_type]
    version = str(args.get("version") or "").strip().upper()
    if version:
        match = [d for d in bouquet if d.version.upper() == version]
        if not match:
            available = ", ".join(d.version for d in bouquet)
            return (
                f"Error: no version {version!r} for {figure_type!r}. "
                f"Available: {available}."
            )
        chosen = [match[0]]
    else:
        chosen = list(bouquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    persist(out_path, figure_type, "(from chat)", chosen, [])
    session.console.print(
        f"[green]✓[/green] Wrote {len(chosen)} figure draft(s) to "
        f"[bold]{out_path}[/bold]"
    )
    return (
        f"Saved {len(chosen)} {figure_type} figure draft(s)"
        + (f" (version {version})" if version else "")
        + f" to {out_path}."
    )


def _save_revision(
    session: ChatSession,
    args: dict[str, Any],
    out_path: Path,
    normalize: Callable[[str], str],
) -> str:
    if not session.recent_revisions:
        return (
            "Error: no revisions in cache. Call revise_draft first to "
            "create one."
        )
    section_raw = str(args.get("section") or "").strip()
    if section_raw:
        try:
            section = normalize(section_raw)
        except ValueError as exc:
            return f"Error: {exc}"
        if section not in session.recent_revisions:
            return (
                f"Error: no cached revision for {section!r}. Available: "
                f"{', '.join(sorted(session.recent_revisions))}."
            )
    elif len(session.recent_revisions) == 1:
        section = next(iter(session.recent_revisions))
    else:
        return (
            "Error: multiple revisions cached "
            f"({', '.join(sorted(session.recent_revisions))}); specify section."
        )
    revision = session.recent_revisions[section]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"# Revised {section}\n\n"
        f"_~{revision.word_count} words "
        f"(target {revision.target_words})_\n\n"
        f"{revision.text}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    session.console.print(
        f"[green]✓[/green] Wrote revision to [bold]{out_path}[/bold]"
    )
    return f"Saved revised {section} to {out_path}."


@register_llm_tool(
    "check_self_plagiarism",
    _function_schema(
        "check_self_plagiarism",
        "Scan a draft against the user's own published-work corpus to "
        "flag accidental self-duplication. Paragraph-level TF-IDF + "
        "cosine similarity, no LLM call. Use after a draft is complete "
        "or when the user worries about overlap with their prior work.",
        {
            "target": {
                "type": "string",
                "description": (
                    "Either a path to a draft file, or a `latest` ref "
                    "('latest', 'latest:<section>', "
                    "'latest:<section>:<version>') resolving against the "
                    "session draft cache."
                ),
            },
            "threshold": {
                "type": "number",
                "description": (
                    "Cosine similarity threshold in (0, 1]. Paragraphs "
                    "at or above this similarity get flagged. Default "
                    "0.4."
                ),
                "minimum": 0.01,
                "maximum": 1.0,
            },
        },
        required=["target"],
    ),
)
def exec_check_self_plagiarism(
    session: ChatSession, args: dict[str, Any]
) -> str:
    from research_agent.cli_check import run_check

    raw_target = str(args.get("target") or "").strip()
    if not raw_target:
        return "Error: target is required."
    threshold_raw = args.get("threshold")
    try:
        threshold = float(threshold_raw) if threshold_raw is not None else 0.4
    except (TypeError, ValueError):
        return "Error: threshold must be a number in (0, 1]."
    if not 0.0 < threshold <= 1.0:
        return "Error: threshold must be in (0, 1]."

    draft_text = ""
    draft_path: Path | None = None
    if raw_target.lower().startswith("latest"):
        resolved = _resolve_latest_draft(session, raw_target)
        if resolved is None:
            return (
                f"Error: could not resolve {raw_target!r} — no cached "
                "drafts. Run draft_section first."
            )
        _, _, draft_text = resolved
    else:
        candidate = Path(raw_target).expanduser()
        if not candidate.exists():
            return f"Error: file not found: {candidate}."
        draft_path = candidate

    try:
        result = run_check(
            session.cfg,
            session.console,
            draft_path=draft_path,
            draft_text=draft_text,
            threshold=threshold,
            output=None,
        )
    except ValueError as exc:
        return f"Error: {exc}"

    report = result.report
    if report.is_clean:
        return (
            f"Self-plagiarism check clean: {report.draft_paragraph_count} "
            f"paragraph(s) scanned, no matches at or above "
            f"{report.threshold:.0%} similarity."
        )
    sample = report.matches[:3]
    lines = [
        f"Self-plagiarism check flagged {len(report.matches)} match(es) at "
        f"or above {report.threshold:.0%} (out of "
        f"{report.draft_paragraph_count} paragraphs).",
        "Top hits:",
    ]
    for m in sample:
        para_excerpt = _truncate(m.draft_paragraph, limit=160)
        lines.append(
            f"- [{m.similarity:.0%}] vs {m.source_id}: {para_excerpt}"
        )
    if len(report.matches) > len(sample):
        lines.append(f"... and {len(report.matches) - len(sample)} more.")
    return "\n".join(lines)


@register_llm_tool(
    "revise_draft",
    _function_schema(
        "revise_draft",
        "Auto-review (Analyst+Critic) then Scribe rewrite. Non-interactive; "
        "revised draft is cached as a revision.",
        {
            "target": {
                "type": "string",
                "description": "Path or `latest` ref to the draft.",
            },
            "section": {
                "type": "string",
                "description": (
                    "Section name. Required when target is a path."
                ),
            },
            "target_words": {
                "type": "integer",
                "description": "Default: match original length.",
                "minimum": 50,
                "maximum": 2000,
            },
        },
        required=["target"],
    ),
)
def exec_revise_draft(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.agents.scribe import normalize_section
    from research_agent.cli_review import run_review

    raw_target = str(args.get("target") or "").strip()
    if not raw_target:
        return "Error: target is required."
    section_raw = str(args.get("section") or "").strip()
    try:
        target_words = int(args.get("target_words") or 0)
    except (TypeError, ValueError):
        return "Error: target_words must be a positive integer."

    draft_text = ""
    draft_path: Path | None = None
    section: str | None = None
    if raw_target.lower().startswith("latest"):
        resolved = _resolve_latest_draft(session, raw_target)
        if resolved is None:
            return (
                f"Error: could not resolve {raw_target!r} — no cached "
                "drafts."
            )
        section, _, draft_text = resolved
    else:
        candidate = Path(raw_target).expanduser()
        if not candidate.exists():
            return f"Error: file not found: {candidate}."
        draft_path = candidate

    if section is None:
        if not section_raw:
            return (
                "Error: section is required when target is a path; pass "
                "section='introduction' (or similar)."
            )
        try:
            section = normalize_section(section_raw)
        except ValueError as exc:
            return f"Error: {exc}"
    elif section_raw:
        # latest ref already supplied section; if user passes one too,
        # normalize it and prefer the explicit value as an override.
        try:
            section = normalize_section(section_raw)
        except ValueError as exc:
            return f"Error: {exc}"

    try:
        result = run_review(
            session.cfg,
            session.console,
            draft_path=draft_path,
            draft_text=draft_text,
            section=section,
            target_words=target_words,
            output=None,
            interactive=False,
            save=False,
        )
    except (ValueError, LLMError) as exc:
        return f"Error from review pipeline: {exc}"

    session.recent_revisions[section] = result.reviewed.revised

    analyst = result.reviewed.analyst_review
    critic = result.reviewed.critic_review
    lines = [
        f"Revised `{section}` via Analyst + Critic + Scribe.",
        f"- Original: {result.reviewed.original.word_count} words.",
        f"- Revised: {result.reviewed.revised.word_count} words.",
    ]
    if analyst is not None:
        lines.append(
            f"- Analyst flagged {len(analyst.issues)} issue(s), "
            f"{len(analyst.suggestions)} suggestion(s)."
        )
    if critic is not None:
        lines.append(
            f"- Critic flagged {len(critic.issues)} issue(s), "
            f"{len(critic.suggestions)} suggestion(s)."
        )
    if analyst and analyst.issues:
        lines.append("Analyst issues:")
        for item in analyst.issues[:5]:
            lines.append(f"  - {_truncate(item, limit=180)}")
    if critic and critic.issues:
        lines.append("Critic issues:")
        for item in critic.issues[:5]:
            lines.append(f"  - {_truncate(item, limit=180)}")
    lines.append(
        "Cached as the latest revision; save with "
        "save_draft_to_file(kind='revision', section='" + section + "', "
        "path='<path>')."
    )
    return "\n".join(lines)


# ---------------------------------------------------------- state mutations
#
# Phase 3 tools surface the write-side CLI commands (style training,
# fingerprint compute/update, config set) as agent-callable functions.
#
# Per the D3 decision: there is NO hard confirmation gate at the tool
# layer. The agent's system prompt instructs the LLM to confirm with
# the user before invoking these tools; if the model bypasses that
# instruction, the worst case is a write the user can rollback (style
# samples can be re-trained, config can be reset). We accept the
# trade-off to keep the tool surface small and code-light.


@register_llm_tool(
    "train_style",
    _function_schema(
        "train_style",
        "Import user writing into the style corpus. State-mutating; "
        "confirm with user first.",
        {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mix of arXiv ids and local PDF paths.",
            },
            "directory": {
                "type": "string",
                "description": (
                    "Folder to scan for .pdf files (non-recursive)."
                ),
            },
            "append": {
                "type": "boolean",
                "description": (
                    "Keep prior samples per paper. Default false (replace)."
                ),
            },
        },
    ),
)
def exec_train_style(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.cli_style import run_style_train

    sources_raw = args.get("sources") or []
    if not isinstance(sources_raw, list):
        return "Error: sources must be an array of strings."
    sources = [str(s).strip() for s in sources_raw if str(s).strip()]
    directory_raw = str(args.get("directory") or "").strip()
    directory: Path | None = (
        Path(directory_raw).expanduser() if directory_raw else None
    )
    append = bool(args.get("append") or False)

    if not sources and directory is None:
        return (
            "Error: provide at least one source (arxiv id, PDF path) or a "
            "directory."
        )

    try:
        result = run_style_train(
            session.cfg,
            session.console,
            sources=sources or None,
            directory=directory,
            replace=not append,
        )
    except Exception as exc:
        return f"Error during style training: {exc}"

    if result.sources_processed == 0 and result.sources_failed == 0:
        return "No sources processed (nothing matched)."
    lines = [
        f"Style training: {result.paragraphs_added} paragraph(s) added "
        f"from {result.sources_processed} source(s); "
        f"{result.sources_failed} failed."
    ]
    if result.per_source:
        for label, count, status in result.per_source[:5]:
            lines.append(f"- {label}: {count} ({status})")
        if len(result.per_source) > 5:
            lines.append(f"... and {len(result.per_source) - 5} more.")
    if result.paragraphs_added > 0:
        lines.append(
            "Next step: build_fingerprint to compute the user's style "
            "vector from these samples."
        )
    return "\n".join(lines)


@register_llm_tool(
    "build_fingerprint",
    _function_schema(
        "build_fingerprint",
        "Compute a style fingerprint from the currently imported samples "
        "and write it to ~/.research-agent/style/fingerprint.json. "
        "Overwrites any existing fingerprint without archiving. "
        "**State-mutating**: confirm with the user before calling, "
        "especially if a fingerprint already exists (use "
        "update_fingerprint instead to preserve history).",
        {},
    ),
)
def exec_build_fingerprint(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.cli_style import run_style_fingerprint

    code = run_style_fingerprint(session.cfg, session.console)
    if code != 0:
        return (
            "Error: no style samples to learn from. Run train_style first."
        )
    return (
        f"Fingerprint built and written to {session.cfg.fingerprint_path}. "
        "Subsequent draft_section / revise_draft calls will use it."
    )


@register_llm_tool(
    "update_fingerprint",
    _function_schema(
        "update_fingerprint",
        "Recompute the style fingerprint from the static sample corpus "
        "PLUS any accepted Scribe revisions, and archive the previous "
        "fingerprint as fingerprint_vN.json so style drift is auditable. "
        "**State-mutating**: confirm with the user before calling.",
        {},
    ),
)
def exec_update_fingerprint(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.cli_style import run_style_update

    code = run_style_update(session.cfg, session.console)
    if code != 0:
        return (
            "Error: no samples or revisions to learn from. Run "
            "train_style first."
        )
    return (
        "Fingerprint updated; the previous version was archived under "
        f"~/.research-agent/style/ for drift inspection. Active "
        f"fingerprint: {session.cfg.fingerprint_path}."
    )


@register_llm_tool(
    "get_config",
    _function_schema(
        "get_config",
        "Read configuration values from ~/.research-agent/config.yaml. "
        "Pass a specific ``key`` to read one field (api_key is always "
        "masked), or omit to get a summary of all keys. Read-only.",
        {
            "key": {
                "type": "string",
                "description": (
                    "Optional. One of: api_key, model, base_url, "
                    "data_dir, app_title, app_url, language, "
                    "alert_threshold."
                ),
            },
        },
    ),
)
def exec_get_config(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.config import KNOWN_KEYS, ConfigError

    key = str(args.get("key") or "").strip().lower()
    cfg = session.cfg
    if key:
        try:
            value = cfg.get_field(key)
        except ConfigError as exc:
            return f"Error: {exc}"
        return f"{key} = {value}"
    lines = ["Current configuration (api_key masked):"]
    for k in sorted(KNOWN_KEYS):
        lines.append(f"- {k} = {cfg.get_field(k)}")
    lines.append(f"- config_path = {cfg.config_path}")
    return "\n".join(lines)


@register_llm_tool(
    "set_config",
    _function_schema(
        "set_config",
        "Update a single configuration field and persist to "
        "~/.research-agent/config.yaml. **State-mutating**: confirm "
        "the exact key + value with the user before calling — this "
        "overwrites disk state and changes to api_key / model / "
        "base_url require a REPL restart to take full effect.",
        {
            "key": {
                "type": "string",
                "description": (
                    "One of: api_key, model, base_url, data_dir, "
                    "app_title, app_url, language, alert_threshold."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "New value (always passed as a string; numeric "
                    "fields like alert_threshold are parsed by the "
                    "config layer)."
                ),
            },
        },
        required=["key", "value"],
    ),
)
def exec_set_config(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.config import ConfigError

    key = str(args.get("key") or "").strip().lower()
    value = args.get("value")
    if not key:
        return "Error: key is required."
    if value is None:
        return "Error: value is required."
    value_str = str(value)
    try:
        session.cfg.set_field(key, value_str)
    except ConfigError as exc:
        return f"Error: {exc}"
    # Mask api_key in the return so the model doesn't echo secrets.
    displayed = (
        session.cfg.masked_api_key()
        if key == "api_key"
        else session.cfg.get_field(key)
    )
    note = ""
    if key in {"api_key", "model", "base_url", "data_dir"}:
        note = (
            " (the change is written to disk; restart the REPL for it "
            "to fully take effect on the active LLM client)"
        )
    return f"Set {key} = {displayed}.{note}"


# ---------------------------------------------------------- diagnostics


@slash(
    "doctor",
    summary="Run environment health checks (config, DB, Chroma, disk).",
    usage="/doctor",
)
def cmd_doctor(session: ChatSession, args: str) -> None:
    from research_agent.cli_doctor import run_doctor

    code = run_doctor(session.cfg, session.console)
    session.memory.append(
        "system",
        "[doctor] "
        + ("All checks passed." if code == 0 else "One or more checks failed."),
    )


@register_llm_tool(
    "run_doctor",
    _function_schema(
        "run_doctor",
        "Run environment health checks: config file, API key, data dir, "
        "SQLite DB integrity, ChromaDB import, disk space, package version. "
        "Read-only, no LLM or network calls. Renders a diagnostic table "
        "to the user and returns a one-line summary for the agent. Call "
        "when the user reports anomalies, asks 'is everything OK', or "
        "before a heavy run.",
        {},
    ),
)
def exec_run_doctor(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.cli_doctor import run_doctor

    code = run_doctor(session.cfg, session.console)
    if code == 0:
        return (
            "Environment health: all checks passed. Diagnostic table "
            "rendered to the user."
        )
    return (
        "Environment health: one or more checks FAILED. Diagnostic table "
        "rendered to the user; suggest fixing the failing rows."
    )


# ---------------------------------------------------------- style (read-only)


@slash(
    "style",
    summary="Show style corpus and fingerprint info (read-only).",
    usage="/style [show|history]",
)
def cmd_style(session: ChatSession, args: str) -> None:
    parts = args.split()
    sub = parts[0].lower() if parts else "show"
    if sub == "show":
        from research_agent.cli_style import run_style_show

        run_style_show(session.cfg, session.console)
        return
    if sub == "history":
        from research_agent.cli_style import run_style_history

        run_style_history(session.cfg, session.console)
        return
    session.console.print(
        f"[yellow]Unknown:[/yellow] /style {sub}. Try /style show|history."
    )


@register_llm_tool(
    "style_show",
    _function_schema(
        "style_show",
        "Show the user's Scribe style corpus and fingerprint summary: "
        "how many paragraphs were imported, from which source papers, "
        "and whether a fingerprint has been built. Read-only. Call when "
        "the user asks 'is my style trained', 'what writing samples have "
        "I imported', or before suggesting a draft/revise action that "
        "needs a fingerprint.",
        {},
    ),
)
def exec_style_show(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.cli_style import run_style_show
    from research_agent.storage.database import Database
    from research_agent.style.samples import StyleSampleRepository

    db = Database(session.cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        total = repo.count()
        by_paper = repo.count_by_paper()
    finally:
        db.close()

    run_style_show(session.cfg, session.console)

    if total == 0:
        return (
            "Style corpus is empty. Suggest the user run train_style "
            "(or `research style train`) to import their writing first."
        )
    fp_path = session.cfg.fingerprint_path
    fp_state = "present" if fp_path.exists() else "NOT built yet"
    return (
        f"Style corpus: {total} paragraph(s) across {len(by_paper)} source "
        f"paper(s). Fingerprint: {fp_state} ({fp_path})."
    )


@register_llm_tool(
    "style_history",
    _function_schema(
        "style_history",
        "List archived fingerprint versions saved under "
        "~/.research-agent/style/. Use when the user wants to see how "
        "their style fingerprint has drifted over time. Read-only.",
        {},
    ),
)
def exec_style_history(session: ChatSession, args: dict[str, Any]) -> str:
    from research_agent.cli_style import run_style_history
    from research_agent.style.fingerprint import Fingerprint

    style_dir = session.cfg.style_dir
    archives = (
        sorted(style_dir.glob("fingerprint_v*.json")) if style_dir.exists() else []
    )
    current = session.cfg.fingerprint_path
    current_version: int | None = None
    if current.exists():
        try:
            current_version = Fingerprint.load_from(current).version
        except (OSError, ValueError):
            current_version = None

    run_style_history(session.cfg, session.console)

    if not archives and current_version is None:
        return (
            "No fingerprint history yet. Suggest the user run "
            "build_fingerprint (or `research style fingerprint`) after "
            "training samples."
        )
    return (
        f"Fingerprint history: {len(archives)} archived version(s); "
        f"current v{current_version if current_version is not None else '?'}. "
        "Table rendered to the user."
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
