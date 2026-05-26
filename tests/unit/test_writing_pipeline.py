"""Unit tests for the writing-review pipeline (M4 S4.3.1)."""

from __future__ import annotations

import json

from research_agent.agents.analyst import Analyst
from research_agent.agents.critic import Critic
from research_agent.agents.orchestrator import Orchestrator
from research_agent.agents.scribe import Draft, Scribe
from research_agent.agents.writing_pipeline import (
    ReviewedDraft,
    WritingReview,
    build_review_prompt,
    build_revision_prompt,
    parse_review,
)
from research_agent.core.llm import MockLLMProvider


def _draft(text: str = "First sentence. Second sentence.") -> Draft:
    return Draft(
        section="introduction",
        version="A",
        variant_label="concise",
        text=text,
        style_note="",
        word_count=len(text.split()),
        target_words=300,
    )


# --- parse_review -----------------------------------------------------------


def test_parse_review_full_payload() -> None:
    raw = json.dumps(
        {
            "issues": ["unsupported claim", "missing baseline"],
            "suggestions": ["cite Kim 2024", "rephrase the lead"],
            "summary": "draft is OK but overclaims",
        }
    )
    review = parse_review("analyst", raw)
    assert review.role == "analyst"
    assert review.issues == ["unsupported claim", "missing baseline"]
    assert review.suggestions == ["cite Kim 2024", "rephrase the lead"]
    assert review.summary == "draft is OK but overclaims"
    assert review.raw_response == raw


def test_parse_review_missing_fields() -> None:
    raw = json.dumps({"issues": ["one"]})
    review = parse_review("critic", raw)
    assert review.role == "critic"
    assert review.issues == ["one"]
    assert review.suggestions == []
    assert review.summary == ""


def test_parse_review_non_json_falls_back_to_summary() -> None:
    review = parse_review("analyst", "plain text reply, no JSON")
    assert review.summary.startswith("plain text reply")
    assert review.issues == []


def test_parse_review_drops_non_string_items() -> None:
    raw = json.dumps({"issues": ["valid", None, "", 42]})
    review = parse_review("analyst", raw)
    assert review.issues == ["valid", "42"]  # None / "" dropped, ints stringified


# --- prompt builders --------------------------------------------------------


def test_build_review_prompt_contains_section_and_text() -> None:
    prompt = build_review_prompt(draft_text="Body.", section="abstract")
    assert "abstract" in prompt
    assert "Body." in prompt
    assert "JSON" in prompt


def test_build_revision_prompt_skips_empty_review() -> None:
    drafted = _draft()
    empty = WritingReview(role="analyst")
    review = WritingReview(role="critic", issues=["overclaim"])
    prompt = build_revision_prompt(draft=drafted, reviews=[empty, review])
    assert "[ORIGINAL DRAFT]" in prompt
    assert "[CRITIC REVIEW]" in prompt
    assert "overclaim" in prompt
    # Empty analyst block should not be rendered
    assert "[ANALYST REVIEW]" not in prompt


# --- ReviewedDraft helpers --------------------------------------------------


def test_reviewed_draft_role_accessors() -> None:
    rd = ReviewedDraft(
        original=_draft(),
        reviews=[
            WritingReview(role="analyst", issues=["a1"]),
            WritingReview(role="critic", issues=["c1"]),
        ],
        revised=_draft("Revised body."),
    )
    assert rd.analyst_review is not None
    assert rd.analyst_review.issues == ["a1"]
    assert rd.critic_review is not None
    assert rd.critic_review.issues == ["c1"]
    assert rd.all_issues() == ["a1", "c1"]


# --- Analyst / Critic review_writing ----------------------------------------


def test_analyst_review_writing_calls_llm() -> None:
    mock = MockLLMProvider()
    mock.enqueue(json.dumps({"issues": ["x"], "suggestions": [], "summary": ""}))
    analyst = Analyst(mock, prompt_stem="analyst_writing")
    review = analyst.review_writing("Some draft text.", "introduction")
    assert review.role == "analyst"
    assert review.issues == ["x"]
    # Confirm the analyst saw an analyst_writing prompt (system msg)
    sys_msg = mock.calls[0][0]
    assert sys_msg.role == "system"
    assert "writing draft" in sys_msg.content.lower()


def test_critic_review_writing_calls_llm() -> None:
    mock = MockLLMProvider()
    mock.enqueue(json.dumps({"issues": ["overclaim"], "summary": "watch the hedges"}))
    critic = Critic(mock, prompt_stem="critic_writing")
    review = critic.review_writing("Draft text.", "results")
    assert review.role == "critic"
    assert review.issues == ["overclaim"]
    assert review.summary == "watch the hedges"


# --- Scribe.revise ----------------------------------------------------------


def test_scribe_revise_returns_revised_draft() -> None:
    mock = MockLLMProvider()
    mock.enqueue(
        json.dumps(
            {
                "draft": "Revised draft body addressing the overclaim.",
                "style_note": "Toned down 'consistently outperforms'.",
            }
        )
    )
    scribe = Scribe(mock)
    drafted = _draft()
    revised = scribe.revise(
        drafted,
        [WritingReview(role="critic", issues=["overclaim"])],
    )
    assert revised.section == drafted.section
    assert revised.version == drafted.version
    assert "Revised draft body" in revised.text
    assert "Toned down" in revised.style_note


def test_scribe_revise_no_actionable_reviews_is_noop() -> None:
    mock = MockLLMProvider()
    scribe = Scribe(mock)
    drafted = _draft()
    revised = scribe.revise(drafted, [WritingReview(role="analyst")])
    assert revised.text == drafted.text
    assert revised.style_note.startswith("No actionable")
    # We should NOT have hit the LLM at all
    assert mock.calls == []


def test_scribe_revise_handles_unparseable_reply() -> None:
    mock = MockLLMProvider()
    mock.enqueue("plain text revision, no JSON")
    scribe = Scribe(mock)
    drafted = _draft()
    revised = scribe.revise(
        drafted,
        [WritingReview(role="critic", issues=["overclaim"])],
    )
    assert "plain text revision" in revised.text


# --- Orchestrator.writing_review_pipeline -----------------------------------


def test_writing_review_pipeline_end_to_end() -> None:
    mock = MockLLMProvider()
    # Two reviews (analyst + critic) + one revision = 3 mock responses
    mock.enqueue(json.dumps({"issues": ["analyst issue"], "summary": "ok"}))
    mock.enqueue(json.dumps({"issues": ["critic issue"], "summary": "watch claims"}))
    mock.enqueue(
        json.dumps(
            {
                "draft": "Revised draft text after addressing reviews.",
                "style_note": "Tightened intro and added baseline mention.",
            }
        )
    )
    orchestrator = Orchestrator(mock)
    scribe = Scribe(mock)
    rd = orchestrator.writing_review_pipeline(scribe, _draft())
    assert isinstance(rd, ReviewedDraft)
    assert rd.analyst_review is not None
    assert rd.analyst_review.issues == ["analyst issue"]
    assert rd.critic_review is not None
    assert rd.critic_review.issues == ["critic issue"]
    assert "Revised draft text" in rd.revised.text
    # Three LLM calls total
    assert len(mock.calls) == 3
