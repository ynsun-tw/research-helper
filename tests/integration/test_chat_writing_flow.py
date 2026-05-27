"""End-to-end happy path: train → build → draft → check → revise → save.

This is the Phase 4 integration test for the R-002 refactor. The unit
tests in tests/unit/test_chat_writing.py and test_chat_state_tools.py
verify each tool independently; this test stitches them together to
confirm that **session state survives the chain**: a draft made via
``draft_section`` is reachable as ``latest:`` from
``check_self_plagiarism`` and ``revise_draft``, and the cached
revision shows up correctly in ``save_draft_to_file``.

The heavy ``run_*`` backends are monkeypatched so we exercise the
tool layer and session caching, not the LLM round-trip itself
(already covered by tests/integration/test_orchestrator.py).
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from research_agent.agents.scribe import Draft
from research_agent.agents.writing_pipeline import ReviewedDraft, WritingReview
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    exec_build_fingerprint,
    exec_check_self_plagiarism,
    exec_draft_section,
    exec_revise_draft,
    exec_save_draft_to_file,
    exec_train_style,
)
from research_agent.cli_review import ReviewResult
from research_agent.cli_style import StyleTrainResult
from research_agent.cli_write import WriteResult
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


@pytest.fixture
def session(tmp_path) -> ChatSession:
    cfg = Config(data_dir=tmp_path, api_key="sk-or-test")
    console = Console(file=StringIO(), width=120)
    return ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )


def test_full_chat_writing_happy_path(
    session: ChatSession, tmp_path: Path, monkeypatch
) -> None:
    # --- Step 1: train_style. Backend mocked — we only care that the
    # tool layer threads the arguments and returns a parseable summary.
    monkeypatch.setattr(
        "research_agent.cli_style.run_style_train",
        lambda cfg, console, **kw: StyleTrainResult(
            sources_processed=2,
            sources_failed=0,
            paragraphs_added=24,
            paragraphs_skipped=0,
            per_source=[
                ("arxiv:2305.14314", 12, "ok"),
                ("arxiv:2301.07041", 12, "ok"),
            ],
        ),
    )
    train_out = exec_train_style(
        session,
        {"sources": ["arxiv:2305.14314", "arxiv:2301.07041"], "append": False},
    )
    assert "24 paragraph" in train_out
    assert "build_fingerprint" in train_out

    # --- Step 2: build_fingerprint. Mocked to "0" since the real
    # implementation needs a populated samples table.
    monkeypatch.setattr(
        "research_agent.cli_style.run_style_fingerprint",
        lambda cfg, console: 0,
    )
    fp_out = exec_build_fingerprint(session, {})
    assert "Fingerprint built" in fp_out

    # --- Step 3: draft_section. Mock run_write to return two variants.
    def fake_run_write(cfg, console, **kw):
        return WriteResult(
            section=kw["section"],
            drafts=[
                Draft(
                    section=kw["section"],
                    version="A",
                    variant_label="concise",
                    text="Concise introduction body. Sparse top-k attention…",
                    style_note="",
                    word_count=8,
                    target_words=kw["target_words"],
                ),
                Draft(
                    section=kw["section"],
                    version="B",
                    variant_label="technical depth",
                    text="Technical introduction body. We propose sparse routing…",
                    style_note="",
                    word_count=8,
                    target_words=kw["target_words"],
                ),
            ],
            context=None,
        )

    monkeypatch.setattr("research_agent.cli_write.run_write", fake_run_write)
    draft_out = exec_draft_section(
        session,
        {
            "section": "intro",
            "context": "sparse top-k attention for 32k contexts",
            "target_words": 300,
            "versions": 2,
        },
    )
    assert "Drafted 2 variant" in draft_out
    # Session cache must be populated under the canonical section name.
    assert "introduction" in session.recent_drafts
    assert len(session.recent_drafts["introduction"]) == 2
    assert session.recent_drafts["introduction"][1].version == "B"

    # --- Step 4: check_self_plagiarism against latest:introduction:B.
    # We never seeded the style_samples table, so the detector should
    # return a clean report — the point of this step is to prove the
    # ``latest:`` ref resolution survives across tool calls.
    check_out = exec_check_self_plagiarism(
        session, {"target": "latest:introduction:B"}
    )
    assert "clean" in check_out.lower()

    # --- Step 5: revise_draft. Mock run_review to return a revised
    # draft + two reviewer reviews. Caches into session.recent_revisions.
    def fake_run_review(cfg, console, **kw):
        original = Draft(
            section=kw["section"],
            version="A",
            variant_label="user input",
            text=kw["draft_text"],
            style_note="",
            word_count=len(kw["draft_text"].split()),
            target_words=kw.get("target_words") or 8,
        )
        revised = Draft(
            section=kw["section"],
            version="R",
            variant_label="revised",
            text="Revised introduction addressing analyst + critic notes.",
            style_note="",
            word_count=7,
            target_words=kw.get("target_words") or 8,
        )
        reviewed = ReviewedDraft(
            original=original,
            reviews=[
                WritingReview(
                    role="analyst",
                    issues=["claim 2 lacks evidence"],
                    suggestions=["cite Tan et al. 2024"],
                ),
                WritingReview(
                    role="critic",
                    issues=["overclaim in last sentence"],
                    suggestions=["hedge the speedup figure"],
                ),
            ],
            revised=revised,
        )
        return ReviewResult(
            reviewed=reviewed, selection=None, saved_revision=None
        )

    monkeypatch.setattr("research_agent.cli_review.run_review", fake_run_review)
    revise_out = exec_revise_draft(
        session, {"target": "latest:introduction"}
    )
    assert "Revised `introduction`" in revise_out
    assert "Analyst flagged 1 issue" in revise_out
    assert "Critic flagged 1 issue" in revise_out
    # The mocked revision text must be the one cached.
    assert "introduction" in session.recent_revisions
    assert session.recent_revisions["introduction"].text.startswith("Revised")

    # --- Step 6: save_draft_to_file pulls the cached revision and
    # writes to disk. No other save call should have hit the FS yet —
    # the integration contract is "drafts only leave memory on demand."
    revision_path = tmp_path / "intro_revised.md"
    save_out = exec_save_draft_to_file(
        session,
        {
            "path": str(revision_path),
            "kind": "revision",
            "section": "introduction",
        },
    )
    assert "Saved revised introduction" in save_out
    assert revision_path.exists()
    body = revision_path.read_text(encoding="utf-8")
    assert "Revised introduction" in body
    assert "addressing analyst + critic notes" in body

    # Also save version B from the original draft cache so we cover the
    # section-save path in the same chain.
    intro_path = tmp_path / "intro_B.md"
    save_b = exec_save_draft_to_file(
        session,
        {
            "path": str(intro_path),
            "kind": "section",
            "section": "introduction",
            "version": "B",
        },
    )
    assert "Saved 1 introduction" in save_b
    assert intro_path.exists()
    assert "Technical introduction body" in intro_path.read_text(encoding="utf-8")

    # Session state is unchanged after saves — the file system is the
    # only thing that should have moved.
    assert len(session.recent_drafts["introduction"]) == 2
    assert "introduction" in session.recent_revisions
