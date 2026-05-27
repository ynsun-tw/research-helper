"""prompt_toolkit-backed input layer for the chat REPL.

Replaces Rich's plain ``console.input`` (which is just a thin wrapper over
Python's ``input()`` and breaks line-editing on terminals where readline
isn't wired up) with a proper ``PromptSession``. This gives us:

* Reliable line editing on every platform (libedit / readline / Windows).
* Cross-session command history at ``cfg.data_dir/repl_history``.
* Tab completion over slash commands.
* Built-in reverse history search (Ctrl+R).
* Ctrl+L to clear the screen.
* A state-aware prompt rendered from ``ChatSession`` (anchor paper, idea).

Tests and headless smoke flows continue to pass ``input_fn=lambda _: ...``
into ``ChatSession.create`` — those skip this module entirely. Production
``run_chat`` constructs a ``PromptSession`` via :func:`build_prompt_session`
and wires it onto ``session.prompt_session``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

if TYPE_CHECKING:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from research_agent.chat.session import ChatSession
    from research_agent.config import Config


_PROMPT_STYLE = Style.from_dict(
    {
        "prompt.label": "bold ansicyan",
        "prompt.state": "ansiyellow",
        "prompt.sep": "ansiwhite",
    }
)


class _SlashCompleter(Completer):
    """Tab-completion for ``/<command>`` tokens.

    Activates only when the buffer starts with a leading slash. Suggests the
    full set of slash command names. We deliberately keep it line-prefix
    based (not whole-line) so users can still complete after typing a few
    characters.
    """

    def __init__(self, command_names: Iterable[str]) -> None:
        self._names = sorted({name for name in command_names if name})

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        # Only suggest while the user is typing the command token itself,
        # i.e. before they've typed a space (after which it's args).
        if " " in text:
            return
        prefix = text[1:].lower()
        for name in self._names:
            if name.startswith(prefix):
                yield Completion(
                    name,
                    start_position=-len(prefix),
                    display=f"/{name}",
                )


def build_prompt_session(
    cfg: Config, command_names: Iterable[str]
) -> PromptSession[str]:
    """Build a ``PromptSession`` wired with history, completion, keybindings.

    History is written to ``cfg.data_dir/repl_history`` (the same place we
    keep config + DB), so it survives across invocations.

    Ctrl+L clears the screen. Ctrl+R (reverse history search) and Ctrl+W
    (word-delete) come for free from prompt_toolkit defaults.
    """
    history_path = Path(cfg.data_dir) / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("c-l")
    def _clear(event):  # type: ignore[no-untyped-def]
        event.app.renderer.clear()

    return PromptSession(
        history=FileHistory(str(history_path)),
        completer=_SlashCompleter(command_names),
        complete_while_typing=False,  # only on Tab; less noisy
        key_bindings=bindings,
        style=_PROMPT_STYLE,
        enable_history_search=True,  # Up arrow does prefix-match history
    )


def render_state_prompt(session: ChatSession) -> FormattedText:
    """Build the per-turn prompt fragments shown by ``PromptSession.prompt``.

    Examples (rendered):

    * ``You> `` — no anchor, no idea.
    * ``You(arxiv:2308.10247)> `` — anchor paper loaded.
    * ``You(arxiv:2308.10247 | idea:sparse-attn)> `` — anchor + active idea.

    The state badge is built from ``ChatSession.anchor_paper`` and
    ``ChatSession.current_idea_id`` so it stays accurate without the router
    needing to track anything extra.
    """
    parts: list[str] = []
    if session.anchor_paper is not None:
        paper_id = session.anchor_paper.id or session.anchor_paper.title[:24]
        parts.append(f"arxiv:{paper_id}")
    if session.current_idea_id:
        # Idea ids can be long uuids; show a short prefix only.
        parts.append(f"idea:{session.current_idea_id[:8]}")

    fragments: list[tuple[str, str]] = [("class:prompt.label", "You")]
    if parts:
        fragments.append(("class:prompt.sep", "("))
        fragments.append(("class:prompt.state", " | ".join(parts)))
        fragments.append(("class:prompt.sep", ")"))
    fragments.append(("class:prompt.label", "> "))
    return FormattedText(fragments)
