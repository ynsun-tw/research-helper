"""Unit tests for :mod:`research_agent.style.plagiarism`."""

from __future__ import annotations

from research_agent.style.plagiarism import (
    ParagraphMatch,
    PlagiarismDetector,
    SimilarityReport,
    _cosine,
    _tfidf_vectors,
    split_into_paragraphs,
    suggest_rewrites,
    tokenize,
)


def test_tokenize_basic() -> None:
    tokens = tokenize("It is OK to write 42 reasonable sentences here.")
    assert "is" not in tokens  # short token (<=2 chars) dropped
    assert "ok" not in tokens  # only 2 chars dropped
    assert "reasonable" in tokens
    assert "sentences" in tokens
    assert "42" not in tokens  # pure-digit tokens not captured (regex requires leading letter)
    assert "write" in tokens


def test_tokenize_preserves_compounds() -> None:
    tokens = tokenize("state-of-the-art")
    # The compound is preserved as a single token
    assert "state-of-the-art" in tokens


def test_split_into_paragraphs_blank_line_break() -> None:
    parts = split_into_paragraphs("First.\n\nSecond.\n\n  \n\nThird.")
    assert parts == ["First.", "Second.", "Third."]


def test_split_into_paragraphs_empty() -> None:
    assert split_into_paragraphs("") == []
    assert split_into_paragraphs("   \n   ") == []


def test_cosine_zero_when_no_overlap() -> None:
    v1 = {"a": 1.0, "b": 2.0}
    v2 = {"c": 1.0, "d": 2.0}
    assert _cosine(v1, v2) == 0.0


def test_cosine_one_for_identical_vectors() -> None:
    v = {"a": 1.0, "b": 1.0}
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_tfidf_handles_empty_docs() -> None:
    out = _tfidf_vectors([[], ["alpha", "beta"]])
    assert out[0] == {}
    assert "alpha" in out[1]
    # IDF can be zero when a term is in all docs, but for a singleton it's positive.
    assert all(v > 0 for v in out[1].values())


# --- PlagiarismDetector -----------------------------------------------------


def test_detector_threshold_validation() -> None:
    for bad in (-0.1, 0.0, 1.5, 2.0):
        try:
            PlagiarismDetector(threshold=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for threshold={bad}")
    PlagiarismDetector(threshold=0.5)  # should not raise


def test_detector_empty_draft_returns_clean_report() -> None:
    det = PlagiarismDetector()
    report = det.check([], [("p1", "Some corpus paragraph here.")])
    assert report.is_clean
    assert report.draft_paragraph_count == 0


def test_detector_empty_corpus_returns_clean_report() -> None:
    det = PlagiarismDetector()
    report = det.check(["Some draft body."], [])
    assert report.is_clean
    assert report.draft_paragraph_count == 1


def test_detector_finds_verbatim_copy() -> None:
    paragraph = (
        "Recurrent neural networks have long dominated sequence modeling "
        "but their sequential nature limits parallelization across the time "
        "dimension."
    )
    det = PlagiarismDetector(threshold=0.4)
    report = det.check(
        [paragraph, "An unrelated paragraph about cooking pasta with olive oil."],
        [("arxiv:0001.0001", paragraph)],
    )
    assert len(report.matches) == 1
    m = report.matches[0]
    assert m.draft_index == 0
    assert m.source_id == "arxiv:0001.0001"
    assert m.similarity > 0.9


def test_detector_skips_below_threshold() -> None:
    det = PlagiarismDetector(threshold=0.95)
    report = det.check(
        ["Recurrent networks dominate sequence modeling."],
        [("p1", "Recurrent attention dominates language modeling.")],
    )
    # Same theme but not identical: should not cross 0.95
    assert report.is_clean


def test_detector_one_match_per_draft_paragraph() -> None:
    p = "Attention is the core ingredient of modern neural network architectures."
    det = PlagiarismDetector(threshold=0.5)
    report = det.check(
        [p],
        [("p1", p), ("p2", p)],
    )
    assert len(report.matches) == 1


def test_detector_sorts_by_similarity_desc() -> None:
    base = "Recurrent neural networks have long dominated sequence modeling."
    near = (
        "Recurrent neural networks have long dominated sequence modeling "
        "with minor variations in detail."
    )
    far = (
        "Self-attention mechanisms operate on the full token grid without "
        "any recurrence whatsoever."
    )
    det = PlagiarismDetector(threshold=0.3)
    report = det.check(
        [base, far],
        [("p1", near), ("p2", far)],
    )
    assert len(report.matches) == 2
    assert report.matches[0].similarity >= report.matches[1].similarity


# --- suggestion + markdown helpers -----------------------------------------


def test_suggest_rewrites_scales_with_similarity() -> None:
    high = ParagraphMatch(0, "d", "s", "s", similarity=0.85)
    mid = ParagraphMatch(0, "d", "s", "s", similarity=0.55)
    low = ParagraphMatch(0, "d", "s", "s", similarity=0.45)
    high_s = suggest_rewrites(high)
    mid_s = suggest_rewrites(mid)
    low_s = suggest_rewrites(low)
    assert any("from scratch" in s for s in high_s)
    assert any("Paraphrase" in s for s in mid_s)
    assert any("Trim" in s or "Cite" in s for s in low_s)


def test_similarity_report_markdown_clean() -> None:
    report = SimilarityReport(threshold=0.4, draft_paragraph_count=5)
    md = report.to_markdown()
    assert "No matches" in md


def test_similarity_report_markdown_with_matches() -> None:
    report = SimilarityReport(
        threshold=0.4,
        draft_paragraph_count=2,
        matches=[
            ParagraphMatch(
                draft_index=0,
                draft_paragraph="Draft body here.",
                source_id="arxiv:0001",
                source_paragraph="Source body here.",
                similarity=0.6,
            ),
        ],
    )
    md = report.to_markdown()
    assert "1 match(es)" in md
    assert "60%" in md
    assert "arxiv:0001" in md
    assert "Paraphrase" in md  # mid-range suggestion
