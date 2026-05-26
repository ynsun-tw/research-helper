"""Alert injection after /read surfaces parked-idea associations (T3.4.1.2).

Unit-level verification that:
- _surface_parked_idea_alerts calls MemoryKeeper.check_associations with
  the paper's title + abstract as context.
- The banner is rendered to the console and recorded in working memory
  so the LLM can pick it up if the user keeps chatting.
- Failures inside check_associations are swallowed (best-effort - the
  user must never see an error mid-read).
- The alert is only emitted when associations come back non-empty.

End-to-end coverage (with the real /read driver + LLM mocks) lands in
T3.4.1.4's integration test.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from research_agent.agents.memory_keeper import Association
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    _alert_threshold,
    _surface_parked_idea_alerts,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper


def _make_session(tmp_path):
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=140)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    return session, console


def _make_paper(title: str = "Attention", abstract: str = "We propose...") -> Paper:
    return Paper(id="arxiv:1706.03762", title=title, abstract=abstract, full_text="")


def _ideas_in_session(session: ChatSession, n: int = 1) -> list[Any]:
    """Create + persist + shelve `n` ideas and return them."""
    out = []
    for i in range(n):
        idea = session.ideas.create(f"parked {i}", f"shelved idea variant {i}")
        session.ideas.update_status(idea.id, "shelved")
        refreshed = session.ideas.get(idea.id)
        assert refreshed is not None
        session.vectors.upsert(refreshed)
        out.append(refreshed)
    return out


def test_surface_alerts_prints_banner_when_associations_found(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    parked = _ideas_in_session(session, 2)

    fake = [
        Association(idea=parked[0], similarity=0.92),
        Association(idea=parked[1], similarity=0.81),
    ]
    captured: dict[str, object] = {}

    def fake_check(self, context, **kw):  # type: ignore[no-untyped-def]
        captured["context"] = context
        captured["kw"] = kw
        return fake

    monkeypatch.setattr(
        "research_agent.agents.memory_keeper.MemoryKeeper.check_associations",
        fake_check,
    )

    paper = _make_paper(title="Transformer caching", abstract="KV cache reuse.")
    try:
        _surface_parked_idea_alerts(session, paper)
        out = console.file.getvalue()
        assert "Related ideas you parked previously" in out
        assert "parked 0" in out
        assert "parked 1" in out
        # Context must contain both title and abstract.
        assert "Transformer caching" in str(captured["context"])
        assert "KV cache reuse" in str(captured["context"])
        # Working memory carries the alert too so the LLM sees it later.
        sys_msgs = [m for m in session.memory.messages if m.role == "system"]
        assert any(
            "parked ideas surfaced" in m.content.lower() for m in sys_msgs
        )
    finally:
        session.close()


def test_surface_alerts_silent_when_no_associations(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    monkeypatch.setattr(
        "research_agent.agents.memory_keeper.MemoryKeeper.check_associations",
        lambda self, context, **kw: [],
    )
    try:
        _surface_parked_idea_alerts(session, _make_paper())
        out = console.file.getvalue()
        # Nothing about "Related ideas..." should appear.
        assert "Related ideas" not in out
        # Nothing extra should land in working memory either.
        sys_msgs = [
            m for m in session.memory.messages if m.role == "system"
        ]
        assert not any("parked ideas surfaced" in m.content.lower() for m in sys_msgs)
    finally:
        session.close()


def test_surface_alerts_swallows_exceptions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misbehaving vector store must not crash /read."""
    session, console = _make_session(tmp_path)

    def boom(self, context, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("chroma is on fire")

    monkeypatch.setattr(
        "research_agent.agents.memory_keeper.MemoryKeeper.check_associations",
        boom,
    )
    try:
        # Must not raise.
        _surface_parked_idea_alerts(session, _make_paper())
        out = console.file.getvalue()
        # And nothing about a chroma error should leak to the user.
        assert "chroma" not in out.lower()
    finally:
        session.close()


def test_surface_alerts_skips_when_paper_text_empty(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty title+abstract -> no vector query at all (saves a call)."""
    session, _ = _make_session(tmp_path)
    called = {"count": 0}

    def counter(self, context, **kw):  # type: ignore[no-untyped-def]
        called["count"] += 1
        return []

    monkeypatch.setattr(
        "research_agent.agents.memory_keeper.MemoryKeeper.check_associations",
        counter,
    )
    try:
        empty_paper = Paper(id="arxiv:x", title="", abstract="", full_text="")
        _surface_parked_idea_alerts(session, empty_paper)
        assert called["count"] == 0
    finally:
        session.close()


def test_alert_threshold_uses_config_value(tmp_path) -> None:
    """The threshold comes from Config.alert_threshold."""
    session, _ = _make_session(tmp_path)
    session.cfg.alert_threshold = 0.65
    try:
        assert _alert_threshold(session) == pytest.approx(0.65)
    finally:
        session.close()


def test_alert_threshold_default_matches_spec(tmp_path) -> None:
    """Fresh Config has alert_threshold == 0.8 (M3 S3.4.1 spec target)."""
    session, _ = _make_session(tmp_path)
    try:
        assert _alert_threshold(session) == pytest.approx(0.8)
    finally:
        session.close()


def test_surface_alerts_forwards_config_threshold(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The threshold from Config flows through to check_associations."""
    session, _ = _make_session(tmp_path)
    session.cfg.alert_threshold = 0.55
    captured: dict[str, object] = {}

    def fake_check(self, context, **kw):  # type: ignore[no-untyped-def]
        captured["threshold"] = kw.get("threshold")
        return []

    monkeypatch.setattr(
        "research_agent.agents.memory_keeper.MemoryKeeper.check_associations",
        fake_check,
    )
    try:
        _surface_parked_idea_alerts(session, _make_paper())
        assert captured["threshold"] == pytest.approx(0.55)
    finally:
        session.close()
