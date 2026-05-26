"""Lock the parse contract for the Pydantic-AI migration (`agents.schemas`).

Each schema must:
* raise ``ValueError`` on JSON garbage (callers depend on this for fallback);
* default missing fields to ``""`` / ``[]`` / type-appropriate zero;
* coerce wrong-shape lists (string, mixed types, ``None`` entries) to
  ``list[str]``;
* clamp out-of-range numeric fields.
"""

from __future__ import annotations

import json

import pytest

from research_agent.agents.schemas import (
    AnalysisPayload,
    ConclusionPayload,
    CritiquePayload,
    DraftPayload,
    FigurePayload,
    IdeaSupportPayload,
    SearcherRefinementPayload,
    SearcherScoresPayload,
    WritingReviewPayload,
    parse_model,
    strip_to_json,
)

# ---------------------------------------------------- strip_to_json envelope


def test_strip_to_json_unwraps_markdown_fence() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert strip_to_json(fenced) == '{"a": 1}'


def test_strip_to_json_slices_prose_around_json() -> None:
    prose = 'Sure, here you go: {"a": 1} hope that helps!'
    assert strip_to_json(prose) == '{"a": 1}'


def test_strip_to_json_passes_through_bare_json() -> None:
    assert strip_to_json('{"a": 1}') == '{"a": 1}'


# ---------------------------------------------------- parse_model error path


def test_parse_model_raises_value_error_on_junk() -> None:
    with pytest.raises(ValueError):
        parse_model("this is not json at all", AnalysisPayload)


def test_parse_model_raises_value_error_on_array_root() -> None:
    # Top-level array is not a valid object envelope.
    with pytest.raises(ValueError):
        parse_model("[1, 2, 3]", AnalysisPayload)


# --------------------------------------------------------- AnalysisPayload


def test_analysis_defaults_for_empty_object() -> None:
    out = parse_model("{}", AnalysisPayload)
    assert out.contributions == []
    assert out.method_insights == []
    assert out.potential_impact == ""
    assert out.related_work == []
    assert out.claimed_vs_evidence == []
    assert out.confidence == 0.5


def test_analysis_clamps_confidence() -> None:
    out = parse_model(json.dumps({"confidence": 5.0}), AnalysisPayload)
    assert out.confidence == 1.0
    out = parse_model(json.dumps({"confidence": -3}), AnalysisPayload)
    assert out.confidence == 0.0
    out = parse_model(json.dumps({"confidence": "bad"}), AnalysisPayload)
    assert out.confidence == 0.5


def test_analysis_coerces_wrong_shape_lists() -> None:
    payload = json.dumps(
        {
            "contributions": "single string instead of list",
            "method_insights": ["valid", None, "", 7],
        }
    )
    out = parse_model(payload, AnalysisPayload)
    assert out.contributions == []
    assert out.method_insights == ["valid", "7"]


def test_analysis_accepts_claim_evidence_pairs() -> None:
    raw = json.dumps(
        {
            "claimed_vs_evidence": [
                {"claim": "C1", "evidence": "E1"},
                "not a dict, should be dropped",
                {"claim": "C2", "evidence": "E2"},
            ]
        }
    )
    out = parse_model(raw, AnalysisPayload)
    assert len(out.claimed_vs_evidence) == 2
    assert out.claimed_vs_evidence[0].claim == "C1"


def test_analysis_ignores_extra_fields() -> None:
    out = parse_model(json.dumps({"hallucinated_field": [1, 2]}), AnalysisPayload)
    assert isinstance(out, AnalysisPayload)


# ---------------------------------------------------- IdeaSupportPayload


def test_idea_support_accepts_assumption_basis_aliases() -> None:
    raw = json.dumps(
        {
            "evidence": [
                {"assumption": "A1", "basis": "B1"},
                {"claim": "C2", "evidence": "E2"},
            ]
        }
    )
    out = parse_model(raw, IdeaSupportPayload)
    assert out.evidence[0].claim == "A1"
    assert out.evidence[0].evidence == "B1"
    assert out.evidence[1].claim == "C2"


# --------------------------------------------------------- ConclusionPayload


def test_conclusion_defaults_to_empty() -> None:
    assert parse_model("{}", ConclusionPayload).conclusion == ""
    out = parse_model(json.dumps({"conclusion": "  done  "}), ConclusionPayload)
    assert out.conclusion == "done"


# ----------------------------------------------------------- CritiquePayload


def test_critique_normalises_score_to_1_9() -> None:
    assert parse_model(json.dumps({"support_score": 10}), CritiquePayload).support_score == 9.0
    assert parse_model(json.dumps({"support_score": 0}), CritiquePayload).support_score == 1.0
    assert parse_model(json.dumps({"support_score": "bad"}), CritiquePayload).support_score == 5.0


def test_critique_accepts_score_aliases() -> None:
    assert parse_model(json.dumps({"score": 7}), CritiquePayload).support_score == 7.0
    assert parse_model(json.dumps({"rating": 6}), CritiquePayload).support_score == 6.0


def test_critique_defaults_to_neutral() -> None:
    out = parse_model("{}", CritiquePayload)
    assert out.support_score == 5.0
    assert out.objections == []
    assert out.suggestions == []


# ---------------------------------------------------- WritingReviewPayload


def test_writing_review_partial_payload() -> None:
    out = parse_model(json.dumps({"issues": ["x"]}), WritingReviewPayload)
    assert out.issues == ["x"]
    assert out.suggestions == []
    assert out.summary == ""


# ------------------------------------------------------------- DraftPayload


def test_draft_strips_whitespace() -> None:
    out = parse_model(json.dumps({"draft": "  body  ", "style_note": "  note  "}), DraftPayload)
    assert out.draft == "body"
    assert out.style_note == "note"


# ---------------------------------------------------- SearcherScoresPayload


def test_searcher_scores_clamps_and_drops_bad_entries() -> None:
    raw = json.dumps(
        {
            "scores": [
                {"index": 1, "score": 0.5, "reason": "a"},
                {"index": "garbage", "score": 0.9, "reason": "drop"},
                {"index": 3, "score": 2.0, "reason": "clamp high"},
                {"index": 4, "score": -0.5, "reason": "clamp low"},
                "not a dict",
            ]
        }
    )
    out = parse_model(raw, SearcherScoresPayload)
    valid_scores = [s for s in out.scores if s.index is not None and s.score is not None]
    assert {s.index for s in valid_scores} == {1, 3, 4}
    by_idx = {s.index: s.score for s in valid_scores}
    assert by_idx[3] == 1.0
    assert by_idx[4] == 0.0


# ------------------------------------------------- SearcherRefinementPayload


def test_refinement_clamps_confidence_and_strips_query() -> None:
    raw = json.dumps({"query": "  cnn  ", "confidence": 7.0})
    out = parse_model(raw, SearcherRefinementPayload)
    assert out.query == "cnn"
    assert out.confidence == 1.0
    assert out.mode is None


def test_refinement_treats_empty_mode_as_none() -> None:
    out = parse_model(json.dumps({"query": "x", "mode": ""}), SearcherRefinementPayload)
    assert out.mode is None


# ------------------------------------------------------------ FigurePayload


def test_figure_partial_payload_fills_blanks() -> None:
    out = parse_model(json.dumps({"code": "\\begin{tikzpicture}\\end{tikzpicture}"}), FigurePayload)
    assert out.code.startswith("\\begin")
    assert out.style_label == ""
    assert out.target_model == ""
