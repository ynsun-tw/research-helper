"""Phase 2 chat writing tools.

Covers the small pure helpers (``_parse_latest_ref``,
``_resolve_latest_draft``, ``_resolve_check_targets``) directly, plus
the high-level executors (``draft_section``, ``draft_figure``,
``save_draft_to_file``, ``check_self_plagiarism``, ``revise_draft``)
with the heavy ``run_*`` backends monkeypatched so we test the
caching + reference-resolution contract without paying for real
LLM calls.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from research_agent.agents.illustrator import FigureDraft
from research_agent.agents.scribe import Draft
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    _parse_latest_ref,
    _resolve_check_targets,
    _resolve_latest_draft,
    exec_check_self_plagiarism,
    exec_draft_figure,
    exec_draft_section,
    exec_revise_draft,
    exec_save_draft_to_file,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider

# ---- fixtures --------------------------------------------------------------


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


def _seed_section(session: ChatSession, section: str = "introduction") -> None:
    session.recent_drafts[section] = [
        Draft(
            section=section,
            version="A",
            variant_label="concise",
            text="First draft body for the introduction section about sparse attention.",
            style_note="generic academic",
            word_count=12,
            target_words=12,
        ),
        Draft(
            section=section,
            version="B",
            variant_label="technical depth",
            text="Second draft body, leaning more technical.",
            style_note="generic academic",
            word_count=7,
            target_words=12,
        ),
    ]


def _seed_figure(session: ChatSession, figure_type: str = "architecture") -> None:
    session.recent_figures[figure_type] = [
        FigureDraft(
            figure_type=figure_type,
            version="A",
            style_label="layered",
            code="\\begin{tikzpicture} ... \\end{tikzpicture}",
            code_language="tikz",
            notes="three encoder blocks stacked",
            suggested_use="paper figure 1",
        )
    ]


# ---- pure helpers ----------------------------------------------------------


def test_parse_latest_ref_variants() -> None:
    assert _parse_latest_ref("latest") == (None, None)
    assert _parse_latest_ref("latest:intro") == ("intro", None)
    assert _parse_latest_ref("latest:intro:b") == ("intro", "B")
    assert _parse_latest_ref("LATEST") == (None, None)
    assert _parse_latest_ref("/path/to/file.md") == (None, None)


def test_resolve_latest_draft_empty_cache(session: ChatSession) -> None:
    assert _resolve_latest_draft(session, "latest") is None


def test_resolve_latest_draft_first_section(session: ChatSession) -> None:
    _seed_section(session)
    resolved = _resolve_latest_draft(session, "latest")
    assert resolved is not None
    section, version, text = resolved
    assert section == "introduction"
    assert version == "A"
    assert "sparse attention" in text


def test_resolve_latest_draft_specific_version(session: ChatSession) -> None:
    _seed_section(session)
    resolved = _resolve_latest_draft(session, "latest:intro:B")
    assert resolved is not None
    _, version, text = resolved
    assert version == "B"
    assert "technical" in text


def test_resolve_latest_draft_unknown_section(session: ChatSession) -> None:
    _seed_section(session)
    assert _resolve_latest_draft(session, "latest:results") is None


def test_resolve_check_targets_mixes_path_and_latest(
    session: ChatSession, tmp_path: Path
) -> None:
    _seed_section(session)
    real_file = tmp_path / "existing.md"
    real_file.write_text("hello world", encoding="utf-8")
    tmpdir = tmp_path / "scratch"
    tmpdir.mkdir()
    paths, errors = _resolve_check_targets(
        session, [str(real_file), "latest:intro:A"], tmpdir
    )
    assert errors == []
    assert len(paths) == 2
    assert paths[0] == real_file
    assert paths[1].parent == tmpdir
    assert "sparse attention" in paths[1].read_text(encoding="utf-8")


def test_resolve_check_targets_reports_errors(
    session: ChatSession, tmp_path: Path
) -> None:
    tmpdir = tmp_path / "scratch"
    tmpdir.mkdir()
    paths, errors = _resolve_check_targets(
        session, ["/nope/missing.md", "latest:results"], tmpdir
    )
    assert paths == []
    assert len(errors) == 2


# ---- draft_section ---------------------------------------------------------


def test_exec_draft_section_caches_and_summarizes(
    session: ChatSession, monkeypatch
) -> None:
    from research_agent.cli_write import WriteResult

    captured: dict[str, object] = {}

    def fake_run_write(cfg, console, **kwargs):
        captured.update(kwargs)
        drafts = [
            Draft(
                section=kwargs["section"],
                version="A",
                variant_label="concise",
                text="Draft text about modular transformers.",
                style_note="",
                word_count=5,
                target_words=kwargs["target_words"],
            )
        ]
        return WriteResult(section=kwargs["section"], drafts=drafts, context=None)

    monkeypatch.setattr("research_agent.cli_write.run_write", fake_run_write)
    out = exec_draft_section(
        session,
        {
            "section": "intro",
            "context": "sparse attention for 32k context",
            "target_words": 150,
            "versions": 1,
        },
    )
    assert "Drafted 1 variant" in out
    assert "introduction" in session.recent_drafts
    assert captured["section"] == "introduction"
    assert captured["target_words"] == 150
    assert captured["versions"] == 1


def test_exec_draft_section_rejects_bad_section(session: ChatSession) -> None:
    out = exec_draft_section(session, {"section": "totally-fake"})
    assert "Error" in out


def test_exec_draft_section_resolves_latest_check_against(
    session: ChatSession, monkeypatch
) -> None:
    _seed_section(session)
    from research_agent.cli_write import WriteResult

    captured_bodies: list[str] = []

    def fake_run_write(cfg, console, **kwargs):
        # Read inside the mock — the real tool wipes its tmpdir in the
        # finally block, so we can't inspect the files after returning.
        for p in kwargs.get("check_against") or []:
            captured_bodies.append(p.read_text(encoding="utf-8"))
        return WriteResult(
            section=kwargs["section"],
            drafts=[
                Draft(
                    section=kwargs["section"],
                    version="A",
                    variant_label="x",
                    text="related work body",
                    style_note="",
                    word_count=3,
                    target_words=10,
                )
            ],
            context=None,
        )

    monkeypatch.setattr("research_agent.cli_write.run_write", fake_run_write)
    out = exec_draft_section(
        session,
        {
            "section": "related_work",
            "check_against": ["latest:introduction:A"],
        },
    )
    assert "Drafted" in out
    assert len(captured_bodies) == 1
    assert "sparse attention" in captured_bodies[0]


# ---- draft_figure ----------------------------------------------------------


def test_exec_draft_figure_caches_and_summarizes(
    session: ChatSession, monkeypatch
) -> None:
    from research_agent.cli_figure import FigureResult

    def fake_run_figure(cfg, console, **kwargs):
        drafts = [
            FigureDraft(
                figure_type=kwargs["figure_type"],
                version="A",
                style_label="layered",
                code="\\begin{tikzpicture}\\end{tikzpicture}",
                code_language="tikz",
                notes="three blocks",
                suggested_use="figure 1",
            )
        ]
        return FigureResult(
            figure_type=kwargs["figure_type"], drafts=drafts, verifications=[]
        )

    monkeypatch.setattr("research_agent.cli_figure.run_figure", fake_run_figure)
    out = exec_draft_figure(
        session,
        {
            "figure_type": "architecture",
            "description": "three-layer encoder",
            "versions": 1,
        },
    )
    assert "Drafted 1 variant" in out
    assert "architecture" in session.recent_figures


def test_exec_draft_figure_requires_description(session: ChatSession) -> None:
    out = exec_draft_figure(
        session, {"figure_type": "architecture", "description": ""}
    )
    assert "description is required" in out


# ---- save_draft_to_file ----------------------------------------------------


def test_save_section_writes_full_bouquet(
    session: ChatSession, tmp_path: Path
) -> None:
    _seed_section(session)
    out_path = tmp_path / "intro.md"
    msg = exec_save_draft_to_file(
        session, {"path": str(out_path), "kind": "section"}
    )
    assert "Saved 2" in msg
    body = out_path.read_text(encoding="utf-8")
    assert "Version A" in body and "Version B" in body


def test_save_section_writes_single_version(
    session: ChatSession, tmp_path: Path
) -> None:
    _seed_section(session)
    out_path = tmp_path / "introB.md"
    msg = exec_save_draft_to_file(
        session,
        {"path": str(out_path), "kind": "section", "version": "B"},
    )
    assert "Saved 1" in msg
    body = out_path.read_text(encoding="utf-8")
    assert "Version B" in body
    assert "Version A" not in body


def test_save_infers_kind_when_unambiguous(
    session: ChatSession, tmp_path: Path
) -> None:
    _seed_section(session)
    out_path = tmp_path / "auto.md"
    msg = exec_save_draft_to_file(session, {"path": str(out_path)})
    assert "Saved" in msg
    assert out_path.exists()


def test_save_errors_when_cache_empty(
    session: ChatSession, tmp_path: Path
) -> None:
    msg = exec_save_draft_to_file(session, {"path": str(tmp_path / "x.md")})
    assert "nothing in the session cache" in msg


def test_save_errors_when_kind_ambiguous(
    session: ChatSession, tmp_path: Path
) -> None:
    _seed_section(session)
    _seed_figure(session)
    msg = exec_save_draft_to_file(session, {"path": str(tmp_path / "x.md")})
    assert "specify kind" in msg


def test_save_unknown_version(session: ChatSession, tmp_path: Path) -> None:
    _seed_section(session)
    msg = exec_save_draft_to_file(
        session,
        {
            "path": str(tmp_path / "x.md"),
            "kind": "section",
            "version": "Z",
        },
    )
    assert "no version" in msg


# ---- check_self_plagiarism -------------------------------------------------


def test_check_self_plagiarism_latest_with_empty_corpus(
    session: ChatSession,
) -> None:
    _seed_section(session)
    # No style samples seeded, so PlagiarismDetector returns is_clean=True.
    out = exec_check_self_plagiarism(session, {"target": "latest:introduction"})
    assert "clean" in out.lower()


def test_check_self_plagiarism_rejects_unresolvable(
    session: ChatSession,
) -> None:
    out = exec_check_self_plagiarism(session, {"target": "latest:nope"})
    assert "could not resolve" in out


def test_check_self_plagiarism_file_path(
    session: ChatSession, tmp_path: Path
) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text("Some draft content for the test.", encoding="utf-8")
    out = exec_check_self_plagiarism(session, {"target": str(draft)})
    assert "clean" in out.lower() or "flagged" in out.lower()


# ---- revise_draft ----------------------------------------------------------


def test_exec_revise_draft_caches_revision(
    session: ChatSession, monkeypatch
) -> None:
    from research_agent.agents.writing_pipeline import (
        ReviewedDraft,
        WritingReview,
    )
    from research_agent.cli_review import ReviewResult

    def fake_run_review(cfg, console, **kwargs):
        original = Draft(
            section=kwargs["section"],
            version="A",
            variant_label="user input",
            text=kwargs.get("draft_text") or "",
            style_note="",
            word_count=5,
            target_words=kwargs.get("target_words") or 5,
        )
        revised = Draft(
            section=kwargs["section"],
            version="R",
            variant_label="revised",
            text="Revised content addressing all issues.",
            style_note="",
            word_count=6,
            target_words=kwargs.get("target_words") or 5,
        )
        reviewed = ReviewedDraft(
            original=original,
            reviews=[
                WritingReview(
                    role="analyst",
                    issues=["weak argumentation"],
                    suggestions=["sharpen claim"],
                ),
                WritingReview(
                    role="critic",
                    issues=["overclaim in conclusion"],
                    suggestions=["hedge it"],
                ),
            ],
            revised=revised,
        )
        return ReviewResult(reviewed=reviewed, selection=None, saved_revision=None)

    monkeypatch.setattr("research_agent.cli_review.run_review", fake_run_review)
    _seed_section(session)
    out = exec_revise_draft(session, {"target": "latest:introduction"})
    assert "Revised" in out
    assert "introduction" in session.recent_revisions
    assert "analyst" in out.lower() or "Analyst" in out
    assert session.recent_revisions["introduction"].text.startswith("Revised")


def test_exec_revise_draft_requires_section_when_path(
    session: ChatSession, tmp_path: Path
) -> None:
    draft = tmp_path / "x.md"
    draft.write_text("hello", encoding="utf-8")
    out = exec_revise_draft(session, {"target": str(draft)})
    assert "section is required" in out


# ---- session field defaults ------------------------------------------------


def test_session_has_writing_caches(session: ChatSession) -> None:
    assert session.recent_drafts == {}
    assert session.recent_figures == {}
    assert session.recent_revisions == {}


# ---- registry --------------------------------------------------------------


def test_phase2_tools_registered() -> None:
    from research_agent.chat.tools import LLM_TOOLS

    for name in (
        "draft_section",
        "draft_figure",
        "save_draft_to_file",
        "check_self_plagiarism",
        "revise_draft",
    ):
        assert name in LLM_TOOLS, f"missing tool: {name}"
