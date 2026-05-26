"""Unit tests for :class:`Scribe` section drafting."""

from __future__ import annotations

import json

from research_agent.agents.scribe import (
    DEFAULT_VARIANTS,
    SECTION_TYPES,
    Scribe,
    _build_user_prompt,
    _format_fingerprint,
    _parse_draft,
    normalize_section,
)
from research_agent.core.llm import MockLLMProvider
from research_agent.style.fingerprint import (
    Fingerprint,
    MacroFingerprint,
    MicroFingerprint,
    PersonalMarkers,
)


def _filled_fp() -> Fingerprint:
    return Fingerprint(
        macro=MacroFingerprint(
            abstract_opener="we propose a",
            intro_opener="recurrent networks have long",
            related_work_strategy="thematic",
            intro_avg_paragraphs=3.0,
        ),
        micro=MicroFingerprint(
            avg_sentence_length=21.0,
            median_sentence_length=20.0,
            p10_sentence_length=10.0,
            p90_sentence_length=33.0,
            avg_paragraph_length=4.5,
            sentence_count=200,
            transition_freq={"however": 2.0, "furthermore": 1.0},
            hedging_per_100=3.2,
            confidence_per_100=1.1,
            passive_per_100=4.0,
            type_token_ratio=0.42,
        ),
        markers=PersonalMarkers(
            top_section_titles=["Introduction", "Method"],
            citation_format="latex_cite",
            figure_ref_format="Figure",
        ),
        sample_count=42,
        paper_count=3,
    )


# --- normalize_section ------------------------------------------------------


def test_normalize_section_canonical() -> None:
    for s in SECTION_TYPES:
        assert normalize_section(s) == s


def test_normalize_section_aliases() -> None:
    assert normalize_section("Intro") == "introduction"
    assert normalize_section("methods") == "method"
    assert normalize_section("RELATED-WORK") == "related_work"
    assert normalize_section("Experiments") == "results"


def test_normalize_section_unknown() -> None:
    try:
        normalize_section("preface")
    except ValueError as exc:
        assert "Unknown section" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_normalize_section_empty() -> None:
    try:
        normalize_section("")
    except ValueError as exc:
        assert "required" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty section")


# --- _format_fingerprint ----------------------------------------------------


def test_format_fingerprint_picks_only_signal() -> None:
    text = _format_fingerprint(_filled_fp())
    assert "abstract opener" in text
    assert "however" in text
    assert "latex_cite" in text


def test_format_fingerprint_empty_signal() -> None:
    text = _format_fingerprint(Fingerprint())
    assert "no actionable signal" in text


# --- _parse_draft -----------------------------------------------------------


def test_parse_draft_json_payload() -> None:
    raw = json.dumps({"draft": "Hello world.", "style_note": "concise"})
    text, note = _parse_draft(raw)
    assert text == "Hello world."
    assert note == "concise"


def test_parse_draft_falls_back_to_raw() -> None:
    raw = "Plain prose draft, no JSON at all."
    text, note = _parse_draft(raw)
    assert text == raw
    assert note == ""


def test_parse_draft_missing_fields() -> None:
    raw = json.dumps({"style_note": "lonely note"})
    text, note = _parse_draft(raw)
    # No draft field → falls back to raw payload.
    assert "lonely note" in text  # raw string contains it
    assert note == "lonely note"


# --- _build_user_prompt -----------------------------------------------------


def test_build_user_prompt_includes_variant_and_words() -> None:
    prompt = _build_user_prompt(
        section="abstract",
        fingerprint=_filled_fp(),
        context="",
        target_words=250,
        variant_label="concise",
        variant_directive="Cut hedges.",
    )
    assert "abstract" in prompt
    assert "250" in prompt
    assert "concise" in prompt
    assert "Cut hedges." in prompt
    assert "Style fingerprint" in prompt


def test_build_user_prompt_handles_no_fingerprint() -> None:
    prompt = _build_user_prompt(
        section="introduction",
        fingerprint=None,
        context="optimizers for low-rank LMs",
        target_words=400,
        variant_label="narrative arc",
        variant_directive="Build the arc.",
    )
    assert "No style fingerprint" in prompt
    assert "optimizers for low-rank LMs" in prompt


# --- Scribe.generate --------------------------------------------------------


def _enqueue_drafts(mock: MockLLMProvider, n: int) -> None:
    for i in range(n):
        mock.enqueue(
            json.dumps(
                {
                    "draft": f"Draft {i} body text spanning several sentences.",
                    "style_note": f"Voice variant {i}.",
                }
            )
        )


def test_generate_three_versions_sequential() -> None:
    mock = MockLLMProvider()
    _enqueue_drafts(mock, 3)
    scribe = Scribe(mock)
    drafts = scribe.generate(
        "abstract",
        fingerprint=_filled_fp(),
        target_words=300,
        n=3,
        parallel=False,
    )
    assert len(drafts) == 3
    versions = [d.version for d in drafts]
    assert versions == ["A", "B", "C"]
    variants = [d.variant_label for d in drafts]
    assert variants == [v[0] for v in DEFAULT_VARIANTS]
    for d in drafts:
        assert d.section == "abstract"
        assert d.target_words == 300
        assert d.word_count > 0
        assert d.text.startswith("Draft ")


def test_generate_cycles_variants_when_n_exceeds_defaults() -> None:
    mock = MockLLMProvider()
    _enqueue_drafts(mock, 5)
    scribe = Scribe(mock)
    drafts = scribe.generate(
        "introduction",
        fingerprint=None,
        n=5,
        parallel=False,
    )
    labels = [d.variant_label for d in drafts]
    expected = [DEFAULT_VARIANTS[i % len(DEFAULT_VARIANTS)][0] for i in range(5)]
    assert labels == expected


def test_generate_zero_versions_returns_empty() -> None:
    scribe = Scribe(MockLLMProvider())
    assert scribe.generate("abstract", n=0, parallel=False) == []


def test_generate_normalizes_section_alias() -> None:
    mock = MockLLMProvider()
    _enqueue_drafts(mock, 1)
    scribe = Scribe(mock)
    drafts = scribe.generate("methods", n=1, parallel=False)
    assert drafts[0].section == "method"


def test_generate_handles_unparseable_llm_reply() -> None:
    mock = MockLLMProvider()
    mock.enqueue("This is a plain text reply with no JSON.")
    scribe = Scribe(mock)
    drafts = scribe.generate("abstract", n=1, parallel=False)
    assert len(drafts) == 1
    assert drafts[0].text.startswith("This is a plain text")
    assert drafts[0].style_note == ""


def test_run_adapter_returns_first_draft() -> None:
    mock = MockLLMProvider()
    _enqueue_drafts(mock, 1)
    scribe = Scribe(mock)
    response = scribe.run({"section": "abstract", "n": 1, "parallel": False})
    assert response.role == "scribe"
    assert response.content.startswith("Draft ")
    drafts = response.metadata.get("drafts")
    assert isinstance(drafts, list)
    assert len(drafts) == 1
