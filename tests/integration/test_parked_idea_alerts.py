"""Integration: parked-idea alerts fire on /read when a related idea exists
(M3 T3.4.1.4).

Constructs a small library:
- One shelved idea with a description that overlaps the next paper's
  abstract on enough distinctive tokens that the Jaccard fallback ranks
  it well above the test-tuned threshold.
- One shelved idea about an unrelated topic (gardening) - must NOT
  appear in the alert banner.

Then runs the real /read flow (via the chat tools' _load_and_analyze
entrypoint) and asserts the related-idea banner is rendered. The vector
store fallback is in-memory Jaccard (use_chroma=False) so no chroma
dependency is needed.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from research_agent.agents.analyst import AnalysisResult
from research_agent.agents.critic import CritiqueResult
from research_agent.agents.orchestrator import AggregatedAnalysis
from research_agent.chat import tools as chat_tools
from research_agent.chat.session import ChatSession
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider
from research_agent.core.paper import Paper, Section


def _make_session(tmp_path: Path) -> tuple[ChatSession, Console]:
    cfg = Config(
        data_dir=tmp_path,
        api_key="sk-x",
        # Jaccard fallback scores are lower than chroma cosine - dial the
        # threshold down for the in-memory backend so this test exercises
        # the *plumbing* (alerts fire end-to-end) rather than the
        # numeric tuning of chroma.
        alert_threshold=0.2,
    )
    console = Console(file=StringIO(), width=160)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    return session, console


def _seed_parked_idea(session: ChatSession, title: str, description: str) -> Any:
    idea = session.ideas.create(title, description)
    session.ideas.update_status(idea.id, "shelved")
    refreshed = session.ideas.get(idea.id)
    assert refreshed is not None
    session.vectors.upsert(refreshed)
    return refreshed


def _stub_orchestrator(session: ChatSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the Analyst+Critic outputs so the integration test focuses on
    the alert path, not the LLM JSON parsing."""
    analyst = AnalysisResult(
        contributions=["transformer KV cache reuse"],
        method_insights=[],
        potential_impact="",
        related_work=[],
        claimed_vs_evidence=[],
        confidence=0.8,
    )
    critic = CritiqueResult(
        objections=[], support_score=7.0, score_reason="ok"
    )
    report = AggregatedAnalysis(analyst=analyst, critic=critic)

    async def fake_analyze(_paper: Paper) -> AggregatedAnalysis:
        return report

    session.orch.analyze_paper_parallel = fake_analyze  # type: ignore[assignment]
    monkeypatch.setattr(chat_tools, "render_paper_header", lambda *a, **k: None)
    monkeypatch.setattr(chat_tools, "render_read_report", lambda *a, **k: None)


def test_read_surfaces_related_parked_idea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    related = _seed_parked_idea(
        session,
        "KV cache reuse for transformer inference",
        "shelved earlier because no compute budget; would speed transformer "
        "inference via KV cache reuse with streaming attention",
    )
    _seed_parked_idea(
        session,
        "Gardening notebook",
        "How to grow heirloom tomatoes in cold-frame winter conditions",
    )

    paper = Paper(
        id="arxiv:1706.03762",
        title="Transformer inference acceleration via KV cache reuse",
        abstract=(
            "We study transformer inference acceleration through KV cache "
            "reuse and streaming attention, sharing cache across requests."
        ),
        sections=[Section(title="Intro", content="speed up inference")],
        full_text="...",
    )

    monkeypatch.setattr(
        chat_tools, "_load_anchor_paper", lambda *a, **k: paper
    )
    _stub_orchestrator(session, monkeypatch)

    try:
        chat_tools._load_and_analyze(session, "1706.03762")
        out = console.file.getvalue()
        # Banner must fire and surface the related idea.
        assert "Related ideas you parked previously" in out
        assert related.title in out
        assert "shelved" in out
        # Unrelated idea must NOT bleed into the banner.
        assert "Gardening notebook" not in out
    finally:
        session.close()


def test_read_skips_alert_when_threshold_filters_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If alert_threshold is high (spec default 0.8 for chroma), the
    keyword-Jaccard fallback's lower-scored matches must NOT fire."""
    cfg = Config(
        data_dir=tmp_path,
        api_key="sk-x",
        alert_threshold=0.85,  # higher than any Jaccard score we'll seed
    )
    console = Console(file=StringIO(), width=160)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    _seed_parked_idea(
        session,
        "KV cache reuse",
        "shelved transformer KV cache reuse idea",
    )
    paper = Paper(
        id="arxiv:1706.03762",
        title="Transformer inference",
        abstract="KV cache reuse for transformers",
        sections=[],
        full_text="",
    )
    monkeypatch.setattr(chat_tools, "_load_anchor_paper", lambda *a, **k: paper)
    _stub_orchestrator(session, monkeypatch)

    try:
        chat_tools._load_and_analyze(session, "1706.03762")
        out = console.file.getvalue()
        # No banner fired.
        assert "Related ideas you parked previously" not in out
    finally:
        session.close()


def test_read_alert_is_silent_when_library_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, console = _make_session(tmp_path)
    paper = Paper(
        id="arxiv:1706.03762",
        title="Transformer inference",
        abstract="KV cache reuse for transformers",
        sections=[],
        full_text="",
    )
    monkeypatch.setattr(chat_tools, "_load_anchor_paper", lambda *a, **k: paper)
    _stub_orchestrator(session, monkeypatch)
    try:
        chat_tools._load_and_analyze(session, "1706.03762")
        out = console.file.getvalue()
        assert "Related ideas you parked previously" not in out
    finally:
        session.close()


def test_read_alert_includes_idea_show_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The banner advertises `/idea show <prefix>` so the user can
    one-key into the related idea's details."""
    session, console = _make_session(tmp_path)
    related = _seed_parked_idea(
        session,
        "KV cache reuse for transformer inference",
        "shelved transformer KV cache reuse for streaming attention inference",
    )
    paper = Paper(
        id="arxiv:1706.03762",
        title="Transformer inference KV cache reuse",
        abstract="KV cache reuse for streaming attention",
        sections=[],
        full_text="",
    )
    monkeypatch.setattr(chat_tools, "_load_anchor_paper", lambda *a, **k: paper)
    _stub_orchestrator(session, monkeypatch)
    try:
        chat_tools._load_and_analyze(session, "1706.03762")
        out = console.file.getvalue()
        # Rich may wrap the banner across lines; check for both the
        # /idea reference and the shortcut id (8-char prefix).
        # Strip whitespace so we don't fight wrapping.
        compact = " ".join(out.split())
        assert "/idea show" in compact
        assert related.id[:8] in compact
    finally:
        session.close()
