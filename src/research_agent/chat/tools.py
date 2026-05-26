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
    summary="Search arXiv by keywords (logs to history; flags already-read).",
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

    session.searches.record(
        args, hits, source="arxiv", session_id=session.memory.session_id
    )
    already_read = session.searches.already_read([h.arxiv_id for h in hits])
    _render_search_hits(session, args, hits, already_read=already_read)
    session.memory.append(
        "system",
        _search_summary(args, hits, already_read=already_read),
    )


def _render_search_hits(
    session: ChatSession,
    query: str,
    hits: list[ArxivSearchHit],
    *,
    already_read: set[str] | None = None,
) -> None:
    already_read = already_read or set()
    table = Table(title=f"arXiv results for: {query}", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Year", justify="right")
    table.add_column("Read", justify="center")
    for hit in hits:
        year = hit.published[:4] if hit.published else "—"
        read_mark = "[green]✓[/green]" if hit.arxiv_id in already_read else ""
        table.add_row(hit.arxiv_id, hit.title[:80], year, read_mark)
    session.console.print(table)
    session.console.print(
        "[dim]Use[/dim] /read <id> [dim]to load one;[/dim] "
        "[green]✓[/green] [dim]= already in your local library.[/dim]"
    )


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
        rows.append(f"- {h.arxiv_id}: {h.title[:80]}{marker}")
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
    session.searches.record(
        query, hits, source="arxiv", session_id=session.memory.session_id
    )
    already_read = session.searches.already_read([h.arxiv_id for h in hits])
    _render_search_hits(session, query, hits, already_read=already_read)
    return _search_summary(query, hits, already_read=already_read)


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
