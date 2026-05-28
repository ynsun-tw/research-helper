"""REPL loop + slash dispatch + LLM agent loop with tool calling."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence

from rich.console import Console
from rich.panel import Panel

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import LLM_TOOLS, SLASH_COMMANDS, llm_tool_schemas
from research_agent.config import Config
from research_agent.core.llm import (
    ChatMessage,
    ChatResponse,
    LLMError,
    LLMProvider,
    ToolCall,
)
from research_agent.memory.working_memory import estimate_tokens

# ``?`` and ``/?`` are conventional REPL shortcuts for "give me help". We
# rewrite them to ``/help`` at the input boundary so users don't have to
# remember the slash form.
HELP_ALIASES = frozenset({"?", "/?", "help"})

# The full tool list lives in the OpenAI ``tools`` payload (see
# ``llm_tool_schemas()``); each schema already carries its own
# ``description``. We deliberately do NOT repeat per-tool prose here —
# duplicating the schemas is what made the earlier prompt ~2.6K tokens.
# Keep this prompt to the *routing rules* the schemas can't express
# (when to chain X→Y, when state-mutation needs confirmation, output
# style). Anything stated here ships on every LLM round-trip.
CHAT_SYSTEM_PROMPT = """\
You are Research Agent, a critical research companion in a CLI shell.

You have function-calling tools for the user's local research state
(papers, debates, ideas, drafts, style corpus). Tool names and
parameters live in the ``tools`` payload — read them; don't fabricate.

Routing rules (schemas don't encode these):
- References to past searches ("that paper from yesterday") → call
  ``recent_searches`` first to recover the arxiv_id, then ``load_paper``.
- References to past conversations ("what did we conclude…") → call
  ``recall_history`` first; briefly cite recalled snippets.
- ``discuss_idea`` requires an anchor paper — call ``load_paper`` first
  if none is loaded.
- "save this for later" / "queue this" → ``queue_add`` (NOT ``load_paper``).
  "read the next one" → ``queue_next`` then ``load_paper`` (use the
  ``source`` hint queue_next returns for local-PDF entries).
- "add my papers folder" / "ingest ~/path" / "queue all PDFs in <dir>" →
  ``ingest_local_papers``. No LLM analysis runs; this just registers
  files for later reading.
- Citation-graph questions ("what cited this?" / "what does this rely on?")
  → ``get_citations`` or ``get_references`` on the anchor paper.
- Environment complaints ("is my setup OK?") → ``run_doctor`` before
  speculating.
- Before ``draft_section`` / ``revise_draft``, if unsure a fingerprint
  exists, call ``style_show`` first. If it doesn't, warn the user the
  draft will use generic academic prose.
- After ``draft_section`` / ``draft_figure``: NEVER auto-save. Paraphrase
  the preview and ask which version + path before calling
  ``save_draft_to_file``.
- "make sure I'm not duplicating my own work" → ``check_self_plagiarism``
  against ``latest:<section>`` or an explicit path.
- "tighten this" / "address the issues" → ``revise_draft``. Surface the
  analyst+critic issue list; don't pretend you applied human judgement
  on each one — the tool is fully automatic.

State-mutating tools (``train_style``, ``build_fingerprint``,
``update_fingerprint``, ``set_config``):
1. Summarise the exact action ("I'll import 12 paragraphs from …",
   "I'll set api_key = sk-or-…").
2. Ask for confirmation.
3. Only call the tool after the user agrees.
4. If irreversible (overwriting an un-archived fingerprint), say so once
   more before calling.

Output:
- Do NOT fabricate paper content or arxiv ids — rely on tool results.
- After tools complete, write a short plain-language summary.
- Be concise. Mirror the user's language.
- Users can also invoke commands directly via ``/search``, ``/read``,
  ``/discuss``, ``/queue``, ``/cites``, ``/refs``, ``/refine``,
  ``/insights``, ``/doctor``, ``/style``, ``/help``, ``/exit``;
  suggest those when a single command is clearer than a tool call.
"""

INTRO_TEXT = (
    "[bold]Research Agent — conversational shell[/bold]\n\n"
    "Type [bold]/help[/bold] or [bold]?[/bold] for commands. Plain text is "
    "sent to the LLM with the current conversation as context.\n"
    "Common flow: [cyan]/search <topic>[/cyan] → [cyan]/read <id>[/cyan] → "
    "[cyan]/discuss <your idea>[/cyan].\n"
    "[dim]Keys: Tab = complete /-commands · ↑/↓ = history · Ctrl+R = "
    "reverse search · Ctrl+L = clear · Ctrl+C = cancel / exit.[/dim]\n"
    "[bold]/exit[/bold] to leave (session is saved)."
)

EXIT_TOKENS = frozenset({"/exit", "/quit", "exit", "quit"})


def run_chat(
    cfg: Config,
    llm: LLMProvider,
    console: Console,
    *,
    input_fn: Callable[[str], str] | None = None,
    use_chroma: bool | None = None,
    opening_message: str | None = None,
) -> int:
    """Enter the conversational REPL. Returns process exit code."""
    # Only build a prompt_toolkit session for real interactive runs. Tests
    # pass an ``input_fn`` lambda; CliRunner-driven e2e tests pipe stdin
    # via stdin which prompt_toolkit can't drive (it needs a real tty for
    # cursor control). In both cases we stay on the plain ``input_fn``
    # path so there's no terminal handle requirement under pytest.
    prompt_session = None
    if input_fn is None and _stdio_is_tty():
        try:
            from research_agent.chat.prompt_ui import build_prompt_session

            prompt_session = build_prompt_session(cfg, SLASH_COMMANDS.keys())
        except Exception:  # pragma: no cover - degrades to console.input
            prompt_session = None

    session = ChatSession.create(
        cfg=cfg,
        llm=llm,
        console=console,
        input_fn=input_fn,
        use_chroma=use_chroma,
        prompt_session=prompt_session,
    )
    console.print(Panel(INTRO_TEXT, border_style="magenta"))

    # Sweep the current working directory for PDFs so the user can launch
    # ``research`` inside a paper folder and immediately see them in
    # ``/queue list``. Already-catalogued files are skipped without re-
    # hashing; net-zero scans stay silent so startup isn't noisier than
    # before.
    from research_agent.chat.tools import _auto_ingest_cwd

    try:
        auto = _auto_ingest_cwd(session)
    except Exception:  # pragma: no cover - never let scan failure break REPL
        auto = None
    if auto is not None and auto.added:
        console.print(
            f"[dim]+ {auto.added} paper(s) from "
            f"[cyan]{auto.folder.name or auto.folder}[/cyan] "
            f"→ reading queue (/queue list)[/dim]"
        )

    # Ctrl+C with an empty buffer no longer kills the session immediately —
    # the user has to press it twice (or once after typing something we
    # cancel). This matches the convention in shells and most modern REPLs
    # so accidental SIGINT doesn't lose unsaved memory.
    ctrl_c_armed = False
    try:
        if opening_message:
            _process_input(session, opening_message.strip())
        while True:
            try:
                raw = _read_user_input(session).strip()
            except KeyboardInterrupt:
                if ctrl_c_armed:
                    console.print(
                        "\n[yellow]Interrupted — saving session…[/yellow]"
                    )
                    break
                console.print(
                    "[dim]Press Ctrl+C again to exit, or type "
                    "/exit.[/dim]"
                )
                ctrl_c_armed = True
                continue
            except EOFError:
                console.print(
                    "\n[yellow]EOF received — saving session…[/yellow]"
                )
                break
            ctrl_c_armed = False
            if not raw:
                continue
            if raw.lower() in EXIT_TOKENS:
                break
            _process_input(session, raw)
    finally:
        saved = session.close()
        console.print(
            f"[green]✓[/green] Session saved "
            f"({session.memory.turn_count()} turn(s), {saved} message(s), "
            f"id={session.memory.session_id[:8]}…)"
        )
    return 0


def _stdio_is_tty() -> bool:
    """True only when both stdin and stdout are real terminals.

    prompt_toolkit needs cursor / key control on stdout and reads keys
    from stdin; piped IO (CliRunner, ``echo … | research``) breaks both,
    so we fall back to plain ``input_fn`` in those cases.
    """
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - stdio closed
        return False


def _read_user_input(session: ChatSession) -> str:
    """Read one line from the user, preferring prompt_toolkit when present.

    Falls back to ``session.input_fn`` (which tests inject) so unit tests
    don't need a real tty. The prompt_toolkit path renders a state-aware
    prompt (anchor paper / active idea) and routes through the configured
    history + completer.
    """
    if session.prompt_session is not None:
        from research_agent.chat.prompt_ui import render_state_prompt

        result = session.prompt_session.prompt(render_state_prompt(session))
        return str(result)
    return session.input_fn("[bold cyan]You>[/bold cyan] ")


def _process_input(session: ChatSession, raw: str) -> None:
    # ``?`` and ``/?`` are help shortcuts users expect from any REPL; rewrite
    # them before dispatch so they never look like LLM prompts.
    if raw.strip().lower() in HELP_ALIASES:
        raw = "/help"
    if raw.startswith("/"):
        _dispatch_slash(session, raw)
    else:
        _chat_with_llm(session, raw)


def _dispatch_slash(session: ChatSession, raw: str) -> None:
    body = raw[1:]
    name, _, args = body.partition(" ")
    name = name.lower().strip()
    cmd = SLASH_COMMANDS.get(name)
    if cmd is None:
        session.console.print(
            f"[yellow]Unknown command:[/yellow] /{name}. Type /help for the list."
        )
        return
    session.memory.append("user", raw)
    try:
        cmd.handler(session, args.strip())
    except LLMError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")


MAX_TOOL_ITERATIONS = 6

# Tool results occasionally include the full text of a paper, a multi-KB
# search hit list, or a draft. Sending those back to the LLM verbatim
# (especially across multiple agent-loop rounds where the same result
# gets re-sent) is the single largest avoidable token sink in this
# system. We cap each ``role=tool`` payload before injecting it into
# ``messages``; the model still sees the head + a tail "(N chars
# elided; ask again with a narrower query if you need the rest)" hint,
# which is enough for it to either work with what's there or ask for a
# narrower call. Roughly 8000 chars ≈ 2000 tokens.
MAX_TOOL_RESULT_CHARS = 8000


def _cap_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    elided = len(text) - MAX_TOOL_RESULT_CHARS
    head = text[:MAX_TOOL_RESULT_CHARS]
    return (
        f"{head}\n\n[... {elided} chars elided to fit context budget. "
        "Re-call the tool with a narrower query / smaller limit / more "
        "specific args if you need the rest.]"
    )


def _chat_with_llm(session: ChatSession, user_text: str) -> None:
    """Run an LLM agent loop: chat → maybe tool_call → execute → loop until text.

    The final, no-more-tool-calls reply is streamed token-by-token via
    ``LLMProvider.chat_with_tools_stream``. Intermediate tool-calling rounds
    also stream any preamble text (the model occasionally says "let me check
    the citations…" before a tool call) and then surface a ``→ calling …``
    line. We deliberately do NOT use ``console.status`` here — the live
    spinner conflicts with progressive token printing.

    Emits a ``[~N in → ~M out tokens · K rounds]`` telemetry line at the
    end so users see the cost shape (and the impact of, e.g., loading a
    huge paper or chaining many tools).
    """
    session.memory.append("user", user_text)
    messages = _build_messages(session)
    tools = llm_tool_schemas() or None
    tools_overhead = _estimate_message_tokens([]) + _estimate_tools_tokens(tools)

    in_tokens = 0
    out_tokens = 0
    rounds = 0

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            rounds += 1
            in_tokens += _estimate_message_tokens(messages) + tools_overhead
            response = _stream_one_response(session, messages, tools)
            out_tokens += estimate_tokens(response.content)
            if not response.has_tool_calls:
                reply = response.content.strip()
                if not reply:
                    session.console.print("[dim](empty reply)[/dim]")
                    session.memory.messages.pop()
                    return
                session.memory.append("assistant", reply)
                _print_token_footer(session, in_tokens, out_tokens, rounds)
                return

            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=tuple(response.tool_calls),
            )
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                result_text = _execute_tool_call(session, tc)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=_cap_tool_result(result_text),
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )
                # Tool results are NOT conversational; persisting them
                # into WorkingMemory used to feed them right back to the
                # LLM on the next turn (via ``_build_messages``), which
                # doubled the cost of every multi-turn session. We tag
                # them with ``kind=tool_log`` so ``_build_messages``
                # filters them out, but keep them in the audit trail and
                # the discussions DB so /history still shows what ran.
                session.memory.append(
                    "system",
                    f"[tool {tc.name}] {result_text[:400]}",
                    metadata={"kind": "tool_log"},
                )

        session.console.print(
            "[yellow]Stopped after too many tool iterations.[/yellow] "
            "Try a more specific request."
        )
        _print_token_footer(session, in_tokens, out_tokens, rounds)
    except LLMError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")


def _estimate_message_tokens(messages: Sequence[ChatMessage]) -> int:
    """Approximate prompt tokens by counting content lengths.

    Uses the same ~4 chars/token heuristic as ``WorkingMemory.estimate_tokens``;
    we don't ship tiktoken so this is intentionally a coarse proxy. Tool-call
    arguments + role overhead are included (~16 tokens per message) so the
    estimate stays in the right order of magnitude for multi-tool turns.
    """
    total = 0
    for m in messages:
        total += estimate_tokens(m.content) + 16
        for tc in m.tool_calls:
            total += estimate_tokens(tc.arguments) + 12
    return total


def _estimate_tools_tokens(tools: list[dict[str, object]] | None) -> int:
    """The tool schemas serialize as JSON on the wire; estimate via dump size."""
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools))


def _print_token_footer(
    session: ChatSession, in_tokens: int, out_tokens: int, rounds: int
) -> None:
    """One-line telemetry so the user can see the cost shape of each turn."""
    session.console.print(
        f"[dim][~{in_tokens} in → ~{out_tokens} out tokens · "
        f"{rounds} round{'s' if rounds != 1 else ''}][/dim]"
    )


def _stream_one_response(
    session: ChatSession,
    messages: list[ChatMessage],
    tools: list[dict[str, object]] | None,
) -> ChatResponse:
    """One round-trip with progressive content output.

    Prints a single ``Thinking…`` cue, then overwrites it with the first
    streamed delta. If the model only emits tool_calls (no content), we
    clear the cue and let the subsequent ``→ calling …`` line take over.
    Returns the full ``ChatResponse`` once the stream completes so the
    caller can inspect ``tool_calls`` and ``content``.
    """
    file = session.console.file
    streamed = {"started": False, "chunks": False}

    file.write("\033[2m…thinking…\033[0m")
    file.flush()

    def on_delta(chunk: str) -> None:
        if not streamed["started"]:
            # First chunk: erase the placeholder ("…thinking…" + reset code).
            # 12 chars of visible text plus the ANSI sequences — overwriting
            # with a generous \r + spaces handles all common terminal widths.
            file.write("\r" + " " * 24 + "\r")
            streamed["started"] = True
        streamed["chunks"] = True
        file.write(chunk)
        file.flush()

    try:
        response = session.llm.chat_with_tools_stream(
            messages,
            tools=tools,
            tool_choice="auto" if tools else "none",
            temperature=0.6,
            on_content_delta=on_delta,
        )
    finally:
        if not streamed["started"]:
            # No content arrived; clear the placeholder before the next print.
            file.write("\r" + " " * 24 + "\r")
            file.flush()
        if streamed["chunks"]:
            # Finish the streamed line with a newline so subsequent Rich
            # output starts at column 0.
            file.write("\n")
            file.flush()
    return response


def _execute_tool_call(session: ChatSession, tc: ToolCall) -> str:
    tool = LLM_TOOLS.get(tc.name)
    if tool is None:
        session.console.print(
            f"[yellow]Model requested unknown tool:[/yellow] {tc.name}"
        )
        return f"Error: tool '{tc.name}' is not available."
    try:
        parsed = json.loads(tc.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Error: invalid arguments JSON: {exc}"
    if not isinstance(parsed, dict):
        return "Error: tool arguments must be a JSON object."

    session.console.print(
        f"[dim]→ calling[/dim] [cyan]{tc.name}[/cyan]"
        + (f" [dim]{_arg_preview(parsed)}[/dim]" if parsed else "")
    )
    try:
        return tool.executor(session, parsed)
    except LLMError as exc:
        return f"Error: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        return f"Error: {exc}"


def _arg_preview(args: dict[str, object]) -> str:
    items = []
    for key, value in list(args.items())[:3]:
        text = str(value)
        if len(text) > 40:
            text = text[:37] + "…"
        items.append(f"{key}={text}")
    return "(" + ", ".join(items) + ")"


def _build_messages(session: ChatSession, *, history_limit: int = 8) -> list[ChatMessage]:
    """Assemble the message array for one LLM call.

    Filters out ``tool_log`` messages: those are persisted for /history
    + audit, but feeding them back to the LLM next turn would be
    double-charging tokens for context the model has already digested
    via the live ``role=tool`` payload during the originating turn.
    The default ``history_limit`` of 8 (was 20) keeps the rolling
    conversation tight; long-range context is recoverable via
    ``recall_history`` on demand.
    """
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
    ]
    if session.anchor_paper is not None:
        paper = session.anchor_paper
        messages.append(
            ChatMessage(
                role="system",
                content=(
                    f"Anchor paper currently loaded: {paper.title} ({paper.id}). "
                    "When discussing this paper, ground claims in its content."
                ),
            )
        )
    eligible = [
        m for m in session.memory.messages
        if m.metadata.get("kind") != "tool_log"
    ]
    for msg in eligible[-history_limit:]:
        if msg.role == "user":
            role: str = "user"
        elif msg.role in {"system", "tool"}:
            role = "system"
        else:
            role = "assistant"
        messages.append(ChatMessage(role=role, content=msg.content))
    return messages
