"""Unit tests for :class:`StyleAnalyzer`."""

from __future__ import annotations

from research_agent.style.analyzer import (
    StyleAnalyzer,
    _classify_section,
    _opener_signature,
    _percentile,
    _pick_format,
)
from research_agent.style.filters import split_sentences
from research_agent.style.samples import StyleSample


def _sample(
    paragraph: str,
    *,
    paper_id: str = "p1",
    section: str = "Introduction",
    sample_id: str = "",
) -> StyleSample:
    sents = split_sentences(paragraph)
    return StyleSample(
        id=sample_id or paragraph[:8],
        paper_id=paper_id,
        section_title=section,
        paragraph=paragraph,
        char_count=len(paragraph),
        word_count=len(paragraph.split()),
        sentence_count=len(sents),
    )


# --- helpers -----------------------------------------------------------------


def test_classify_section_buckets() -> None:
    assert _classify_section("Abstract") == "abstract"
    assert _classify_section("1 Introduction") == "intro"
    assert _classify_section("Related Work") == "related"
    assert _classify_section("Background") == "related"
    assert _classify_section("Prior work") == "related"
    assert _classify_section("Method") == "other"
    assert _classify_section("") == "other"


def test_opener_signature() -> None:
    assert _opener_signature("We propose a new model for X.") == "we propose a new model"
    assert _opener_signature("") == ""


def test_percentile() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7.0], 0.5) == 7.0
    # Linear interpolation: 10/90 of [1..10]
    assert abs(_percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.10) - 1.9) < 1e-9
    assert abs(_percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.90) - 9.1) < 1e-9


def test_pick_format_dominant_and_mixed() -> None:
    assert _pick_format({"a": 0, "b": 0}) == ""
    assert _pick_format({"a": 10, "b": 0, "c": 0}) == "a"
    assert _pick_format({"a": 5, "b": 5}) == "mixed"
    # Too little signal for "mixed"
    assert _pick_format({"a": 1, "b": 1}) == ""


# --- end-to-end --------------------------------------------------------------


def test_analyze_empty_returns_default_fingerprint() -> None:
    fp = StyleAnalyzer().analyze([])
    assert fp.sample_count == 0
    assert fp.paper_count == 0
    assert fp.micro.avg_sentence_length == 0.0


def test_analyze_micro_basic_stats() -> None:
    p1 = (
        "Recurrent networks have long dominated sequence modeling. "
        "However, the rise of attention has reshaped the field. "
        "We propose to push this trend further."
    )
    p2 = (
        "Specifically, our model removes recurrence entirely. "
        "We may overstate the impact, but the trend is unmistakable. "
        "Furthermore, our results clearly demonstrate gains."
    )
    samples = [_sample(p1), _sample(p2)]
    fp = StyleAnalyzer().analyze(samples)
    assert fp.sample_count == 2
    assert fp.paper_count == 1
    assert fp.micro.sentence_count == 6
    assert fp.micro.avg_sentence_length > 0
    assert fp.micro.p10_sentence_length <= fp.micro.avg_sentence_length
    assert fp.micro.p90_sentence_length >= fp.micro.avg_sentence_length
    assert fp.micro.avg_paragraph_length == 3.0
    # Word lists should each count at least one hit
    assert "however" in fp.micro.transition_freq
    assert "furthermore" in fp.micro.transition_freq
    assert "specifically" in fp.micro.transition_freq
    assert fp.micro.hedging_per_100 > 0  # "may"
    assert fp.micro.confidence_per_100 > 0  # "clearly", "demonstrate"
    assert 0.0 < fp.micro.type_token_ratio < 1.0


def test_analyze_macro_uses_section_buckets() -> None:
    abstract = _sample(
        "We propose a new attention-only architecture. It outperforms recurrent baselines. "
        "The approach is simple and trainable.",
        section="Abstract",
        paper_id="pA",
    )
    intro_a = _sample(
        "Recurrent networks have long dominated sequence modeling. "
        "Their sequential nature limits parallelization. We aim to overcome that.",
        section="Introduction",
        paper_id="pA",
    )
    intro_b = _sample(
        "Recurrent networks have long dominated sequence modeling. "
        "Their sequential nature limits parallelization. We aim to overcome that.",
        section="Introduction",
        paper_id="pB",
    )
    related = _sample(
        "Prior work in 2014 and 2017 explored attention. "
        "Several approaches followed in 2019 with different goals. "
        "A subsequent line of work in 2021 unified these directions.",
        section="Related Work",
        paper_id="pA",
    )
    fp = StyleAnalyzer().analyze([abstract, intro_a, intro_b, related])
    assert fp.paper_count == 2
    assert fp.macro.abstract_opener.startswith("we propose")
    assert fp.macro.intro_opener.startswith("recurrent networks")
    assert fp.macro.abstract_avg_sentences > 0
    assert fp.macro.related_work_strategy == "chronological"
    # 'Introduction' appears in two papers + abstract / related = 1 each => section avg
    assert fp.macro.section_count_avg >= 1.0


def test_extract_markers_detects_dominant_citation_style() -> None:
    p1 = _sample(
        "We follow prior work \\cite{Smith2020} on this topic. "
        "Further details \\cite{Brown2021} are available. "
        "More references \\cite{Doe2019} appear later."
    )
    fp = StyleAnalyzer().analyze([p1])
    assert fp.markers.citation_format == "latex_cite"


def test_extract_markers_detects_mixed_citation_style() -> None:
    p1 = _sample(
        "Earlier studies [1, 2] and modern work \\cite{Smith2020} both apply. "
        "Author-year citations (Brown, 2021) also appear. "
        "Multiple bracket forms [3] coexist."
    )
    fp = StyleAnalyzer().analyze([p1])
    assert fp.markers.citation_format == "mixed"


def test_extract_markers_picks_figure_abbr_when_dominant() -> None:
    p1 = _sample(
        "See Fig. 1 for the architecture. Fig. 2 reports ablations. "
        "Fig. 3 compares baselines."
    )
    fp = StyleAnalyzer().analyze([p1])
    assert fp.markers.figure_ref_format == "Fig."


def test_extract_markers_empty_when_no_signals() -> None:
    p1 = _sample(
        "We discuss findings without referencing anything formally here. "
        "The paragraph contains no figures or citations. "
        "Just prose carries the argument."
    )
    fp = StyleAnalyzer().analyze([p1])
    assert fp.markers.citation_format == ""
    assert fp.markers.figure_ref_format == ""


def test_top_section_titles_ordered_by_frequency() -> None:
    samples = [
        _sample("First.", section="Introduction"),
        _sample("Second.", section="Introduction"),
        _sample("Third.", section="Method"),
    ]
    fp = StyleAnalyzer().analyze(samples)
    assert fp.markers.top_section_titles[0] == "Introduction"
