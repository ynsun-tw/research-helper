"""Activation-condition alerts on search hits (M3 T3.4.2.3).

Unit-level coverage for ``_surface_activation_alerts``: given a session
with shelved/waiting ideas that carry user-supplied
``activation_conditions``, the helper scans incoming search hits and
emits a banner if any condition phrase appears (case-insensitively) in
``hit.title + hit.abstract``.

The matcher is intentionally simple (substring) - rationale documented
in the helper docstring and the milestone notes. These tests pin the
contract so we can swap in a smarter matcher later without losing
behaviour guarantees.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import _surface_activation_alerts
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.search.arxiv_search import ArxivSearchHit


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


def _hit(arxiv_id: str, title: str, abstract: str = "") -> ArxivSearchHit:
    return ArxivSearchHit(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        published="2024-01-01",
    )


def _make_shelved_idea_with_condition(session: ChatSession, title: str, condition: str) -> str:
    idea = session.ideas.create(title=title, description="…")
    session.ideas.update_status(idea.id, "shelved")
    session.ideas.add_activation_condition(idea.id, condition)
    return idea.id


def _console_out(session: ChatSession) -> str:
    file = session.console.file
    assert isinstance(file, StringIO)
    return file.getvalue()


def test_banner_fires_when_hit_title_contains_condition(session) -> None:
    idea_id = _make_shelved_idea_with_condition(
        session, "Train SLM on FineWeb-Edu", "FineWeb-Edu dataset"
    )
    hits = [
        _hit("2401.0001", "Announcing the FineWeb-Edu Dataset", "abstract"),
        _hit("2401.0002", "Unrelated study on transformers", "abstract"),
    ]

    _surface_activation_alerts(session, hits)
    out = _console_out(session)
    compact = " ".join(out.split())
    assert "Shelved idea(s) may have an unblock" in compact
    assert "2401.0001" in compact
    assert "FineWeb-Edu dataset" in compact
    assert "Train SLM on FineWeb-Edu" in compact
    # The 8-char prefix shortcut must appear so the user can jump straight in.
    assert idea_id[:8] in compact
    # The non-matching hit must not be in the banner.
    assert "2401.0002" not in compact


def test_banner_matches_against_abstract_too(session) -> None:
    _make_shelved_idea_with_condition(
        session, "Idea", "Llama-3.5 release"
    )
    hits = [
        _hit(
            "2401.0010",
            "A study on instruction tuning",
            "We use the recently released Llama-3.5 release as our base model.",
        )
    ]
    _surface_activation_alerts(session, hits)
    assert "2401.0010" in _console_out(session)


def test_case_insensitive_match(session) -> None:
    _make_shelved_idea_with_condition(session, "Idea", "FINEWEB DATASET")
    hits = [_hit("2401.0020", "Releasing fineweb dataset v2", "")]
    _surface_activation_alerts(session, hits)
    assert "2401.0020" in _console_out(session)


def test_silent_when_no_ideas_have_conditions(session) -> None:
    idea = session.ideas.create("No-condition idea", "…")
    session.ideas.update_status(idea.id, "shelved")
    hits = [_hit("2401.0001", "Title with FineWeb-Edu Dataset", "")]
    _surface_activation_alerts(session, hits)
    assert "Shelved idea(s)" not in _console_out(session)


def test_silent_for_ideas_with_inactive_statuses(session) -> None:
    # Active idea (not shelved/waiting): conditions are ignored.
    idea = session.ideas.create("Active idea", "…")
    session.ideas.add_activation_condition(idea.id, "FineWeb")
    hits = [_hit("2401.0001", "FineWeb dataset paper", "")]
    _surface_activation_alerts(session, hits)
    assert "Shelved idea(s)" not in _console_out(session)


def test_no_hits_short_circuits(session) -> None:
    _make_shelved_idea_with_condition(session, "Idea", "anything")
    _surface_activation_alerts(session, [])
    assert "Shelved idea(s)" not in _console_out(session)


def test_no_matching_condition(session) -> None:
    _make_shelved_idea_with_condition(session, "Idea", "Falcon-180B release")
    hits = [_hit("2401.0001", "GPT improvements", "no falcon here")]
    _surface_activation_alerts(session, hits)
    assert "Shelved idea(s)" not in _console_out(session)


def test_banner_caps_at_three_matches(session) -> None:
    _make_shelved_idea_with_condition(session, "Idea", "kw")
    hits = [_hit(f"2401.{i:04d}", "Title with kw inside", "abs") for i in range(5)]
    _surface_activation_alerts(session, hits)
    out = _console_out(session)
    # All five match, but only 3 lines should render.
    assert out.count("matches condition") == 3


def test_only_first_matching_condition_per_idea_hit(session) -> None:
    idea = session.ideas.create("Idea", "")
    session.ideas.update_status(idea.id, "shelved")
    session.ideas.add_activation_condition(idea.id, "dataset A")
    session.ideas.add_activation_condition(idea.id, "dataset B")
    hits = [_hit("2401.0001", "Paper using dataset A and dataset B", "")]
    _surface_activation_alerts(session, hits)
    out = _console_out(session)
    # Should only render one bullet, not two.
    assert out.count("2401.0001") == 1


def test_exceptions_are_swallowed(monkeypatch, session) -> None:
    """The matcher must never crash a /search even if downstream blows up."""

    def boom(*a, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("idea store is broken")

    monkeypatch.setattr(session.ideas, "list_all", boom)
    hits = [_hit("2401.0001", "Anything", "")]
    # Must not raise:
    _surface_activation_alerts(session, hits)


def test_match_writes_system_memory(session) -> None:
    _make_shelved_idea_with_condition(session, "SLM on FineWeb", "FineWeb-Edu dataset")
    hits = [_hit("2401.0001", "Released FineWeb-Edu dataset paper", "")]
    _surface_activation_alerts(session, hits)
    sys_msgs = [m for m in session.memory.messages if m.role == "system"]
    assert any("activation conditions" in m.content for m in sys_msgs)
    assert any("SLM on FineWeb" in m.content for m in sys_msgs)
