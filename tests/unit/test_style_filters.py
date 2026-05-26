"""Unit tests for paragraph filters in :mod:`research_agent.style.filters`."""

from __future__ import annotations

from research_agent.style.filters import (
    is_useful_paragraph,
    section_is_blocked,
    split_sentences,
)


def test_split_sentences_simple() -> None:
    sents = split_sentences("First sentence. Second one! Third? Last.")
    assert len(sents) == 4
    assert sents[0] == "First sentence"


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("    ") == []


def test_section_is_blocked() -> None:
    assert section_is_blocked("References")
    assert section_is_blocked("4. references")
    assert section_is_blocked("Acknowledgements")
    assert section_is_blocked("Acknowledgments")
    assert section_is_blocked("Appendix A")
    assert section_is_blocked("Supplementary Material")
    assert not section_is_blocked("Introduction")
    assert not section_is_blocked("Method")
    assert not section_is_blocked("")


def test_useful_paragraph_accepts_prose() -> None:
    text = (
        "Recurrent neural networks have long dominated sequence modeling "
        "for natural language tasks. However, their sequential nature "
        "limits parallelization across the time dimension. This bottleneck "
        "motivates the architectures we describe below."
    )
    assert is_useful_paragraph(text)


def test_useful_paragraph_rejects_short() -> None:
    assert not is_useful_paragraph("Too short.")
    assert not is_useful_paragraph("")


def test_useful_paragraph_rejects_single_sentence_caption() -> None:
    # Long enough char-wise but only one sentence - typical figure caption.
    text = (
        "Figure 1: A diagram showing the multi-head self-attention "
        "computation pathway and its associated parameter shapes"
    )
    assert not is_useful_paragraph(text)


def test_useful_paragraph_rejects_math_dense() -> None:
    text = (
        "We compute x = w_1 * h + w_2 * h + b where 0 < w_i < 1 and "
        "sum_i w_i = 1. Then y = sigma(z) for z = x * 2 + 3 = 5."
    )
    assert not is_useful_paragraph(text)


def test_useful_paragraph_rejects_too_long() -> None:
    # Construct a 5000+ char string of plausible prose.
    chunk = (
        "Some prose about machine learning systems and their failure modes. "
        "We elaborate further on each component in turn. "
    )
    big = chunk * 50  # well past MAX_CHARS
    assert not is_useful_paragraph(big)


def test_useful_paragraph_rejects_low_letter_density() -> None:
    # Lots of digits/punctuation - simulates a table dump.
    text = "1.23 | 4.56 | 7.89 | 0.12 | 3.45 | 6.78 | 9.01 | 2.34 | 5.67 | 8.90 |"
    text *= 3
    assert not is_useful_paragraph(text)
