"""Unit tests for paragraph splitting + sample extraction."""

from __future__ import annotations

from pathlib import Path

from research_agent.core.paper import Paper, Section
from research_agent.style.extractor import extract_samples, split_paragraphs


def _prose(seed: str = "A") -> str:
    return (
        f"{seed}ttention has emerged as a core ingredient of modern neural "
        "networks. We argue that it is not merely a complement to recurrence "
        "but a strict replacement. The remainder of this section motivates "
        "this claim with three observations."
    )


def test_split_paragraphs_blank_line_break() -> None:
    body = "First paragraph here is fine.\n\nSecond paragraph stands alone.\n"
    parts = split_paragraphs(body)
    assert len(parts) == 2
    assert parts[0].startswith("First")
    assert parts[1].startswith("Second")


def test_split_paragraphs_heals_hyphenation() -> None:
    body = "We recogni-\nze the pattern across all benchmarks."
    parts = split_paragraphs(body)
    assert parts == ["We recognize the pattern across all benchmarks."]


def test_split_paragraphs_soft_breaks_collapsed() -> None:
    # PyMuPDF style: every visual line ends in a newline even mid-sentence.
    body = (
        "Recurrent neural networks have long\n"
        "dominated sequence modeling and we\n"
        "challenge that assumption here."
    )
    parts = split_paragraphs(body)
    # Should be one logical paragraph after collapsing soft breaks.
    assert len(parts) == 1
    assert "challenge that assumption" in parts[0]


def test_extract_samples_uses_sections_and_filters() -> None:
    paper = Paper(
        id="arxiv:0000.0001",
        title="A paper",
        authors=["Smith"],
        abstract="",
        sections=[
            Section(title="Introduction", content=_prose("A") + "\n\n" + _prose("B")),
            Section(title="References", content=_prose("X") + "\n\n" + _prose("Y")),
        ],
        full_text="",
        pdf_path=Path("/tmp/x.pdf"),
    )
    samples = extract_samples(paper)
    # 2 paragraphs from Introduction; References section blocked wholesale.
    assert len(samples) == 2
    assert all(s.section_title == "Introduction" for s in samples)
    assert all(s.paper_id == "arxiv:0000.0001" for s in samples)
    for s in samples:
        assert s.char_count > 0
        assert s.word_count > 0
        assert s.sentence_count > 0


def test_extract_samples_falls_back_to_full_text() -> None:
    paper = Paper(
        id="local:abc",
        title="A paper",
        authors=[],
        abstract="",
        sections=[],
        full_text=_prose("A") + "\n\n" + _prose("B"),
        pdf_path=Path("/tmp/x.pdf"),
    )
    samples = extract_samples(paper)
    assert len(samples) == 2
    assert all(s.section_title == "" for s in samples)


def test_extract_samples_drops_short_and_math_paragraphs() -> None:
    paper = Paper(
        id="local:abc",
        title="A paper",
        authors=[],
        abstract="",
        sections=[
            Section(title="Introduction", content="Too short.\n\n" + _prose("A")),
            Section(
                title="Method",
                content=(
                    "We compute x = w_1 * h + b. Then y = sigma(z) for z = x * 2 + 3 = 5. "
                    "And w = 1.0 + 2.0 + 3.0 = 6.0 with i = 1 and j = 2."
                ),
            ),
        ],
        full_text="",
        pdf_path=Path("/tmp/x.pdf"),
    )
    samples = extract_samples(paper)
    # Only the prose paragraph from Introduction survives.
    assert len(samples) == 1
    assert samples[0].section_title == "Introduction"
