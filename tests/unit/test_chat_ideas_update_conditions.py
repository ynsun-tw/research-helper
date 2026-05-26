"""`/ideas update --condition` slash parsing (M3 T3.4.2.2).

Verifies the slash command extracts `--condition` (greedy across spaces),
allows multiple `--condition` flags per invocation, and supports
`--clear-conditions` to wipe the list. Other update modes
(`--status`, `--feedback`) keep working alongside.

`run_ideas_update` (the CLI handler) is exercised via the chat slash
which delegates into it, so the user-visible contract is covered without
mocking the CLI surface.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import cmd_ideas
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


@pytest.fixture
def session(tmp_path):
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=140)
    return ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )


def test_single_condition_added(session) -> None:
    idea = session.ideas.create("Some idea", "…")
    cmd_ideas(session, f"update {idea.id[:8]} --condition FineWeb-Edu dataset release")
    refreshed = session.ideas.get(idea.id)
    assert refreshed is not None
    assert refreshed.activation_conditions == ["FineWeb-Edu dataset release"]


def test_multiple_conditions_in_one_command(session) -> None:
    idea = session.ideas.create("Idea", "")
    cmd_ideas(
        session,
        f"update {idea.id[:8]} "
        f"--condition needs FineWeb dataset "
        f"--condition needs 1B checkpoint",
    )
    refreshed = session.ideas.get(idea.id)
    assert refreshed is not None
    assert refreshed.activation_conditions == [
        "needs FineWeb dataset",
        "needs 1B checkpoint",
    ]


def test_clear_conditions(session) -> None:
    idea = session.ideas.create("Idea", "")
    session.ideas.add_activation_condition(idea.id, "something")
    cmd_ideas(session, f"update {idea.id[:8]} --clear-conditions")
    refreshed = session.ideas.get(idea.id)
    assert refreshed is not None
    assert refreshed.activation_conditions == []


def test_clear_then_set_in_one_command(session) -> None:
    idea = session.ideas.create("Idea", "")
    session.ideas.add_activation_condition(idea.id, "old condition")
    cmd_ideas(
        session,
        f"update {idea.id[:8]} --clear-conditions --condition shiny new condition",
    )
    refreshed = session.ideas.get(idea.id)
    assert refreshed is not None
    assert refreshed.activation_conditions == ["shiny new condition"]


def test_status_and_condition_together(session) -> None:
    idea = session.ideas.create("Idea", "")
    cmd_ideas(
        session,
        f"update {idea.id[:8]} --status shelved --condition release Llama-3.5",
    )
    refreshed = session.ideas.get(idea.id)
    assert refreshed is not None
    assert refreshed.status == "shelved"
    assert refreshed.activation_conditions == ["release Llama-3.5"]


def test_unknown_idea_id_reports_error(session) -> None:
    cmd_ideas(session, "update deadbeef --condition foo")
    assert "Idea not found" in session.console.file.getvalue()


def test_no_changes_prints_usage_hint(session) -> None:
    idea = session.ideas.create("Idea", "")
    cmd_ideas(session, f"update {idea.id[:8]}")
    out = session.console.file.getvalue()
    assert "Nothing to update" in out
