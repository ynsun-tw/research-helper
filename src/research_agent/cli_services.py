"""CLI command implementations (testable without Typer runner)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from research_agent.agents.analyst import AnalysisResult
from research_agent.agents.critic import CritiqueResult
from research_agent.agents.debate import DebateHistory, DebateResult, FollowUpResult
from research_agent.agents.memory_keeper import MemoryKeeper
from research_agent.agents.orchestrator import AggregatedAnalysis, Orchestrator
from research_agent.config import Config
from research_agent.core.idea import Idea
from research_agent.core.llm import LLMError, LLMProvider
from research_agent.core.loader import PaperLoadError, load_paper
from research_agent.core.paper import Paper
from research_agent.core.paper_resolver import (
    load_paper_from_hit,
    search_arxiv_papers,
    select_arxiv_hit,
    try_direct_paper_load,
)
from research_agent.memory.working_memory import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_CONTEXT_TOKENS,
    WorkingMemory,
)
from research_agent.storage.database import Database, PaperRepository
from research_agent.storage.discussions import DiscussionRepository
from research_agent.storage.ideas import IdeaRepository
from research_agent.storage.vector_store import IdeaVectorStore
from research_agent.ui.formatting import (
    render_debate_result,
    render_followup_turn,
    render_paper_header,
    render_read_report,
)


def _load_anchor_paper(
    paper_query: str,
    *,
    cfg: Config,
    console: Console,
    input_fn: Callable[[str], str],
) -> Paper:
    """Search/load anchor paper; interactive selection runs outside Rich status spinners."""
    with console.status("[bold]Loading anchor paper…[/bold]"):
        direct = try_direct_paper_load(paper_query, cache_dir=cfg.pdf_cache_dir)
    if direct is not None:
        console.print(f"[green]✓[/green] Loaded [bold]{direct.title}[/bold]")
        return direct

    with console.status("[bold]Searching arXiv…[/bold]"):
        hits = search_arxiv_papers(paper_query)

    if len(hits) == 1:
        console.print(
            f"[green]✓[/green] Found [bold]{hits[0].title}[/bold] "
            f"([cyan]{hits[0].arxiv_id}[/cyan])"
        )
        chosen = hits[0]
    else:
        chosen = select_arxiv_hit(hits, console=console, input_fn=input_fn)

    with console.status(f"[bold]Downloading {chosen.arxiv_id}…[/bold]"):
        paper = load_paper_from_hit(chosen, cache_dir=cfg.pdf_cache_dir)
    console.print(f"[green]✓[/green] Loaded [bold]{paper.title}[/bold]")
    return paper


def run_read(
    source: str,
    cfg: Config,
    llm: LLMProvider,
    console: Console,
) -> int:
    """Load a paper, analyze with Analyst + Critic, render and persist."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            load_task = progress.add_task("Loading paper…", total=None)
            try:
                paper = load_paper(source, cache_dir=cfg.pdf_cache_dir)
            except PaperLoadError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                return 1
            progress.update(load_task, description="[green]✓[/green] Paper loaded")

            parse_task = progress.add_task("Parsing PDF…", total=None)
            progress.update(parse_task, description="[green]✓[/green] PDF parsed")

            analyze_task = progress.add_task("Analyzing (Analyst + Critic)…", total=None)
            orch = Orchestrator(llm, language=cfg.language)
            report = asyncio.run(orch.analyze_paper_parallel(paper))
            progress.update(analyze_task, description="[green]✓[/green] Analysis complete")

            save_task = progress.add_task("Saving results…", total=None)
            _persist_read_results(cfg, paper, report)
            progress.update(save_task, description="[green]✓[/green] Results saved")

        render_read_report(console, paper, report)
        return 0
    except LLMError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    except ValueError as exc:
        console.print(f"[red]Error:[/red] Could not parse model output: {exc}")
        return 1


def run_discuss(
    cfg: Config,
    llm: LLMProvider,
    console: Console,
    *,
    paper_query: str,
    opening_topic: str | None = None,
    idea_id: str | None = None,
    input_fn: Callable[[str], str] | None = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    use_chroma: bool | None = None,
    prompt_save_idea: bool = True,
) -> int:
    """Interactive paper-anchored Idea debate REPL."""
    if input_fn is None:
        input_fn = console.input

    memory = WorkingMemory.new_session()
    memory.idea_id = idea_id
    debate = DebateHistory(idea_id=idea_id)
    idea_seed = (opening_topic or "").strip()

    db = Database(cfg.db_path)
    discussions = DiscussionRepository(db)
    papers = PaperRepository(db)
    ideas = IdeaRepository(db)
    if use_chroma is not None:
        chroma = use_chroma
    else:
        chroma = not bool(os.environ.get("RESEARCH_AGENT_TEST_MODE"))
    vectors = IdeaVectorStore(cfg.chroma_dir, use_chroma=chroma)
    keeper = MemoryKeeper(ideas, vectors)
    orch = Orchestrator(llm, language=cfg.language)

    try:
        anchor_paper = _load_anchor_paper(
            paper_query,
            cfg=cfg,
            console=console,
            input_fn=input_fn,
        )
    except PaperLoadError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        db.close()
        return 1

    papers.save(anchor_paper)
    render_paper_header(console, anchor_paper)

    if idea_id:
        linked = ideas.get(idea_id)
        if linked is None:
            console.print(
                f"[yellow]Warning:[/yellow] Idea id {idea_id} not found; starting new debate."
            )
        else:
            idea_seed = idea_seed or linked.description or linked.title

    console.print(
        Panel(
            "All debate is grounded in the anchor paper above.\n"
            "First message: your idea [italic]about this paper[/italic] → supports, "
            "objections, suggestions, score.\n"
            "Later messages: conclusions from paper + debate context (no repeat scoring).\n"
            "Prefix [bold]@analyst[/bold] or [bold]@critic[/bold] to direct a reply.\n"
            "Type [bold]feedback: note[/bold] to record score disagreement (no score change).\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to leave (session is saved).",
            title="[bold]Research Discuss[/bold]",
            border_style="magenta",
        )
    )

    if idea_seed:
        reminder = keeper.format_reminders(
            keeper.recall_similar(idea_seed, exclude_id=idea_id),
            query_text=idea_seed,
        )
        if reminder:
            console.print(Panel(reminder, border_style="yellow"))
    elif not opening_topic:
        console.print(
            "[dim]Enter your research idea about this paper "
            "(how to extend, apply, or critique it).[/dim]"
        )

    saved_idea: Idea | None = None

    try:
        if opening_topic:
            code = _handle_debate_turn(
                opening_topic,
                idea_seed,
                debate,
                memory,
                orch,
                console,
                paper=anchor_paper,
                ideas=ideas,
                max_context_tokens=max_context_tokens,
            )
            if code != 0:
                return code

        while True:
            try:
                user_input = input_fn("[bold cyan]You>[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Interrupted — saving session…[/yellow]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            if user_input.lower().startswith("feedback:"):
                note = user_input.split(":", 1)[1].strip()
                if memory.idea_id and note:
                    ideas.add_user_score_feedback(memory.idea_id, note)
                    console.print("[green]✓[/green] Feedback recorded.")
                elif note:
                    console.print("[yellow]Save this idea first to attach feedback.[/yellow]")
                continue

            if not idea_seed:
                idea_seed = user_input

            code = _handle_debate_turn(
                user_input,
                idea_seed,
                debate,
                memory,
                orch,
                console,
                paper=anchor_paper,
                ideas=ideas,
                max_context_tokens=max_context_tokens,
            )
            if code != 0:
                return code

    finally:
        saved = memory.persist(discussions)
        db.close()
        console.print(
            f"[green]✓[/green] Session saved "
            f"({memory.turn_count()} turn(s), {saved} message(s), id={memory.session_id[:8]}…)"
        )
        if prompt_save_idea and debate.rounds and memory.idea_id is None:
            saved_idea = _prompt_save_idea(
                console,
                input_fn,
                ideas,
                vectors,
                idea_seed,
                debate,
                memory,
                anchor_paper=anchor_paper,
            )
        elif debate.rounds and memory.idea_id:
            idea = ideas.get(memory.idea_id)
            if idea is not None:
                vectors.upsert(idea)

    if saved_idea is not None:
        sid = saved_idea.id[:8]
        console.print(f"[green]✓[/green] Idea saved: [bold]{saved_idea.title}[/bold] ({sid}…)")
    return 0


def _handle_debate_turn(
    user_input: str,
    idea_seed: str,
    debate: DebateHistory,
    memory: WorkingMemory,
    orch: Orchestrator,
    console: Console,
    *,
    paper: Paper,
    ideas: IdeaRepository,
    max_context_tokens: int,
) -> int:
    message, target = _parse_agent_target(user_input)
    memory.append("user", user_input)
    context = _build_debate_context(debate, max_context_tokens)

    if not debate.has_initial_round:
        return _handle_initial_debate_turn(
            message,
            idea_seed,
            debate,
            memory,
            orch,
            console,
            paper=paper,
            ideas=ideas,
            context=context,
            target=target,
        )
    return _handle_followup_debate_turn(
        message,
        idea_seed,
        debate,
        memory,
        orch,
        console,
        paper=paper,
        context=context,
        target=target,
    )


def _handle_initial_debate_turn(
    message: str,
    idea_seed: str,
    debate: DebateHistory,
    memory: WorkingMemory,
    orch: Orchestrator,
    console: Console,
    *,
    paper: Paper,
    ideas: IdeaRepository,
    context: str,
    target: str | None,
) -> int:
    try:
        with console.status("[bold]Opening debate…[/bold] (Analyst + Critic)"):
            result = asyncio.run(
                orch.debate_round_async(
                    idea_seed,
                    context,
                    paper=paper,
                    history=debate,
                    target=target,
                )
            )
    except LLMError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        memory.messages.pop()
        return 0
    except (ValueError, KeyError, TypeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        memory.messages.pop()
        return 0

    render_debate_result(console, result)
    debate.append(message, result, targeted_agent=target)

    if memory.idea_id:
        ideas.append_score(
            memory.idea_id,
            result.score,
            result.score_reason,
            session_id=memory.session_id,
        )
        idea = ideas.get(memory.idea_id)
        if idea is not None:
            idea.critic_objections = list(result.objections)
            ideas.save(idea)

    memory.append(
        "analyst",
        _supports_text(result),
        metadata={"debate_round": result.round_index, "target": target, "phase": "initial"},
    )
    memory.append(
        "critic",
        _critique_text(result),
        metadata={"score": result.score, "debate_round": result.round_index, "phase": "initial"},
    )
    return 0


def _handle_followup_debate_turn(
    message: str,
    idea_seed: str,
    debate: DebateHistory,
    memory: WorkingMemory,
    orch: Orchestrator,
    console: Console,
    *,
    paper: Paper,
    context: str,
    target: str | None,
) -> int:
    try:
        with console.status("[bold]Replying…[/bold]"):
            followup = asyncio.run(
                orch.followup_turn_async(
                    idea_seed,
                    message,
                    context,
                    paper=paper,
                    target=target,
                )
            )
    except LLMError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        memory.messages.pop()
        return 0
    except (ValueError, KeyError, TypeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        memory.messages.pop()
        return 0

    render_followup_turn(console, followup)
    debate.append_followup(message, followup, targeted_agent=target)
    _persist_followup_memory(memory, followup, target=target)
    return 0


def _persist_followup_memory(
    memory: WorkingMemory,
    followup: FollowUpResult,
    *,
    target: str | None,
) -> None:
    if followup.analyst_conclusion:
        memory.append(
            "analyst",
            followup.analyst_conclusion,
            metadata={"phase": "followup", "target": target},
        )
    if followup.critic_conclusion:
        memory.append(
            "critic",
            followup.critic_conclusion,
            metadata={"phase": "followup", "target": target},
        )


def _parse_agent_target(user_input: str) -> tuple[str, str | None]:
    lowered = user_input.strip().lower()
    for tag in ("@critic", "@analyst"):
        if lowered.startswith(tag):
            rest = user_input[len(tag) :].strip()
            return rest or user_input, tag[1:]
    return user_input, None


def _build_debate_context(history: DebateHistory, max_tokens: int) -> str:
    if not history.rounds:
        return ""
    chunks: list[str] = []
    for i, rnd in enumerate(history.rounds, start=1):
        if rnd.result is not None:
            chunks.append(
                f"Round {i} (opening) — User: {rnd.user_message}\n{rnd.result.to_markdown()}"
            )
        elif rnd.followup is not None:
            parts = [f"Round {i} (follow-up) — User: {rnd.user_message}"]
            if rnd.followup.analyst_conclusion:
                parts.append(f"Analyst: {rnd.followup.analyst_conclusion}")
            if rnd.followup.critic_conclusion:
                parts.append(f"Critic: {rnd.followup.critic_conclusion}")
            chunks.append("\n".join(parts))
    text = "\n\n".join(chunks)
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _supports_text(result: object) -> str:
    from research_agent.agents.debate import DebateResult

    assert isinstance(result, DebateResult)
    lines = ["## Supports", *[f"- {s}" for s in result.supports]]
    if result.suggestions:
        lines.extend(["", "## Suggestions", *[f"- {s}" for s in result.suggestions]])
    return "\n".join(lines)


def _critique_text(result: object) -> str:
    from research_agent.agents.debate import DebateResult

    assert isinstance(result, DebateResult)
    obj_lines = [f"- {o}" for o in result.objections] or ["- (none)"]
    lines = [
        f"## Support score: {result.score:.0f}/9",
        f"**Reason:** {result.score_reason or '(not provided)'}",
        "",
        "## Objections",
        *obj_lines,
    ]
    return "\n".join(lines)


def _prompt_save_idea(
    console: Console,
    input_fn: Callable[[str], str],
    ideas: IdeaRepository,
    vectors: IdeaVectorStore,
    idea_seed: str,
    debate: DebateHistory,
    memory: WorkingMemory,
    *,
    anchor_paper: Paper,
) -> Idea | None:
    try:
        answer = input_fn("Save as Idea? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None
    if answer not in {"y", "yes"}:
        return None

    title = idea_seed.split("\n", 1)[0][:80] or "Untitled idea"
    try:
        title_input = input_fn(f"Idea title [{title}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        title_input = ""
    if title_input:
        title = title_input

    idea = ideas.create(title=title, description=idea_seed)
    idea.related_papers = [anchor_paper.id]
    ideas.save(idea)
    memory.idea_id = idea.id
    debate.idea_id = idea.id

    opening: DebateResult | None = None
    for rnd in reversed(debate.rounds):
        if rnd.result is not None:
            opening = rnd.result
            break
    if opening is not None:
        idea.critic_score = opening.score
        idea.critic_objections = list(opening.objections)
        ideas.append_score(
            idea.id,
            opening.score,
            opening.score_reason,
            session_id=memory.session_id,
        )
        ideas.save(idea)

    vectors.upsert(idea)
    return idea


def _persist_read_results(cfg: Config, paper: Paper, report: AggregatedAnalysis) -> None:
    db = Database(cfg.db_path)
    try:
        repo = PaperRepository(db)
        repo.save(paper)
        repo.save_analysis_notes(
            paper.id,
            analyst_notes=_analysis_to_dict(report.analyst),
            critic_notes=_critique_to_dict(report.critic),
        )
    finally:
        db.close()


def _analysis_to_dict(result: AnalysisResult) -> dict[str, object]:
    from dataclasses import asdict

    data = asdict(result)
    data["claimed_vs_evidence"] = [
        {"claim": p.claim, "evidence": p.evidence} for p in result.claimed_vs_evidence
    ]
    return data


def _critique_to_dict(result: CritiqueResult) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(result)
