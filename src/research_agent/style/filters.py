"""Paragraph-level filters that drop non-prose content from style samples.

Style training only wants the *user's own narrative writing*: the kind of
text that reveals their voice (sentence rhythm, transition vocabulary,
hedging style). The PDF parser is greedy and returns everything,
including reference lists, formula-dense methodology pages, table
captions, footnote glue, and section headings stuck to body text.

This module is the gate that keeps that junk out of the
``style_samples`` table. We deliberately use cheap text heuristics
instead of an LLM call - thousands of paragraphs per training run
times one model call each adds up fast, and the failure mode for an
over-eager filter is benign (we'll just have fewer samples; the
training set is supposed to be small and high-signal anyway).
"""

from __future__ import annotations

import re

# A section title that contains any of these tokens is filtered wholesale -
# every paragraph inside it gets dropped without further inspection.
_BLOCKED_SECTION_TOKENS: tuple[str, ...] = (
    "references",
    "bibliography",
    "acknowledg",  # acknowledgement / acknowledgment
    "appendix",
    "supplementary",
    "footnotes",
)

# Paragraphs shorter than this many characters are noise (page headers,
# stray figure captions, footnotes). 80 chars ~ one tweet-length
# sentence, which is the smallest piece of prose with any signal.
MIN_CHARS = 80
# Above this size we're almost certainly looking at a multi-page run-on
# from a PDF parse glitch - the paragraph might be useful but it's also
# almost impossible to weight cleanly against shorter samples, so we
# drop it.
MAX_CHARS = 4000

# Same for sentence count - one-sentence "paragraphs" are usually
# captions or section subtitles glued onto the body.
MIN_SENTENCES = 2

# A rough sentence splitter. Good enough for English research papers;
# we explicitly don't try to be clever about abbreviations because the
# downstream Fingerprint analyzer just needs a count, not perfect spans.
_SENTENCE_END = re.compile(r"[.!?](?:\s+|$)")


def split_sentences(text: str) -> list[str]:
    """Return the (rough) list of sentences inside a paragraph.

    Used by both the filter and the micro-fingerprint analyzer in S4.1.2,
    so it lives here as a single source of truth. Splits on
    ``[.!?]\\s+`` and drops the empties.
    """
    parts = _SENTENCE_END.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def section_is_blocked(section_title: str) -> bool:
    """Return True if every paragraph under this section should be dropped."""
    lowered = (section_title or "").lower()
    return any(token in lowered for token in _BLOCKED_SECTION_TOKENS)


def is_useful_paragraph(text: str) -> bool:
    """Decide whether a single paragraph is worth keeping as a style sample.

    The hierarchy of checks (cheapest first):
    1. Length guards - skip anything too short or absurdly long.
    2. Sentence count - reject single-sentence captions.
    3. Math density - reject anything where math/symbol chars dominate
       (formula-heavy methodology pages tell us nothing about prose voice).
    4. Letter density - reject anything where < 60% of non-whitespace
       characters are letters (table dumps, ASCII figures, citation
       bibliographies that slipped past the section filter).
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < MIN_CHARS or len(stripped) > MAX_CHARS:
        return False
    if len(split_sentences(stripped)) < MIN_SENTENCES:
        return False
    if _math_density(stripped) > 0.10:
        return False
    return _letter_density(stripped) >= 0.60


def _math_density(text: str) -> float:
    """Fraction of characters that look like math.

    Counts the obvious symbols (``= + - * / ^ < > ≤ ≥ ≠ ± ∑ ∫ ∂``) plus
    digits. Anything > 10% almost certainly indicates a method block
    with inline formulas rather than narrative prose.
    """
    if not text:
        return 0.0
    math_chars = sum(1 for ch in text if ch in "=+-*/^<>≤≥≠±∑∫∂∇∞" or ch.isdigit())
    return math_chars / max(1, len(text))


def _letter_density(text: str) -> float:
    """Fraction of non-whitespace characters that are alphabetic."""
    if not text:
        return 0.0
    non_ws = [ch for ch in text if not ch.isspace()]
    if not non_ws:
        return 0.0
    letters = sum(1 for ch in non_ws if ch.isalpha())
    return letters / len(non_ws)
