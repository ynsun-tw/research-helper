"""REPL loop + slash dispatch + LLM agent loop with tool calling."""

from __future__ import annotations

import json
from collections.abc import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import LLM_TOOLS, SLASH_COMMANDS, llm_tool_schemas
from research_agent.config import Config
from research_agent.core.llm import ChatMessage, LLMError, LLMProvider, ToolCall

CHAT_SYSTEM_PROMPT = """\
You are Research Agent, a critical research companion running inside a CLI shell.

You have access to function-calling tools that operate on the user's local state
(papers, debates, saved ideas). Call them when the request needs real data:

  search_arxiv(query, max_results?)   - find candidate papers on arXiv
  recent_searches(limit?)             - look up the user's past /search history
                                        and which arXiv ids they already read;
                                        use this to resolve references like
                                        "the BERT paper from yesterday"
  recall_history(query, limit?)       - semantic search across past REPL
                                        discussions (cross-session); use this
                                        when the user references something
                                        previously discussed
  load_paper(source)                  - download + analyze (Analyst + Critic);
                                        sets the anchor paper
  discuss_idea(idea)                  - debate an idea grounded in the anchor
                                        paper (must load_paper first)
  save_current_idea(title?)           - persist the active debate as an idea
  list_ideas()                        - list saved ideas

Rules:
- When the user refers back to an earlier search ("that transformer paper
  I searched last week"), call recent_searches first to recover the
  arxiv_id, then chain into load_paper.
- When the user refers to a prior conversation ("we already talked about X",
  "what did we conclude about Y"), call recall_history first; cite the
  recalled snippets briefly so the user sees the connection.
- Always call load_paper before discuss_idea; if no paper is loaded, do it first.
- Do NOT fabricate paper content or arXiv ids; rely on tool results.
- After tools complete, write a short, plain-language summary for the user.
- The user can also invoke commands directly with slashes (/search, /history,
  /recall, /read, /discuss, /paper, /idea, /ideas, /help, /exit) - mention
  those when guidance is more useful than a tool call.
- Be concise. Mirror the user's language.
"""

INTRO_TEXT = (
    "[bold]Research Agent — conversational shell[/bold]\n\n"
    "Type [bold]/help[/bold] for commands. Plain text is sent to the LLM with "
    "the current conversation as context.\n"
    "Common flow: [cyan]/search <topic>[/cyan] → [cyan]/read <id>[/cyan] → "
    "[cyan]/discuss <your idea>[/cyan].\n"
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
    session = ChatSession.create(
        cfg=cfg,
        llm=llm,
        console=console,
        input_fn=input_fn,
        use_chroma=use_chroma,
    )
    console.print(Panel(INTRO_TEXT, border_style="magenta"))

    try:
        if opening_message:
            _process_input(session, opening_message.strip())
        while True:
            try:
                raw = session.input_fn("[bold cyan]You>[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Interrupted — saving session…[/yellow]")
                break
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


def _process_input(session: ChatSession, raw: str) -> None:
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


def _chat_with_llm(session: ChatSession, user_text: str) -> None:
    """Run an LLM agent loop: chat → maybe tool_call → execute → loop until text."""
    session.memory.append("user", user_text)
    messages = _build_messages(session)
    tools = llm_tool_schemas() or None

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            with session.console.status("[bold]Thinking…[/bold]"):
                response = session.llm.chat_with_tools(
                    messages,
                    tools=tools,
                    tool_choice="auto" if tools else "none",
                    temperature=0.6,
                )
            if not response.has_tool_calls:
                reply = response.content.strip()
                if not reply:
                    session.console.print("[dim](empty reply)[/dim]")
                    session.memory.messages.pop()
                    return
                session.console.print(Markdown(reply))
                session.memory.append("assistant", reply)
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
                        content=result_text,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )
                session.memory.append(
                    "system",
                    f"[tool {tc.name}] {result_text[:400]}",
                )

        session.console.print(
            "[yellow]Stopped after too many tool iterations.[/yellow] "
            "Try a more specific request."
        )
    except LLMError as exc:
        session.console.print(f"[red]Error:[/red] {exc}")


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


def _build_messages(session: ChatSession, *, history_limit: int = 20) -> list[ChatMessage]:
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
    for msg in session.memory.messages[-history_limit:]:
        if msg.role == "user":
            role: str = "user"
        elif msg.role in {"system", "tool"}:
            role = "system"
        else:
            role = "assistant"
        messages.append(ChatMessage(role=role, content=msg.content))
    return messages
