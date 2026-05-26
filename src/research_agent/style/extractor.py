"""Turn a :class:`Paper` into a list of :class:`StyleSample` rows.

The PDF parser already splits the document into sections. Our job is
to split each *body* section into paragraphs, run the filter pipeline,
and emit a sample row for everything that survives.

Paragraph detection in PDF-extracted text is famously messy. We treat
two-or-more newlines as the canonical paragraph break, but PyMuPDF
often inserts only a single newline at the end of every visual line.
To compensate we also greedy-join consecutive single-line breaks when
the previous line ends in a sentence-final character.
"""

from __future__ import annotations

import re

from research_agent.core.paper import Paper
from research_agent.style.filters import (
    is_useful_paragraph,
    section_is_blocked,
    split_sentences,
)
from research_agent.style.samples import StyleSample

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_HYPHEN_LINE_BREAK = re.compile(r"-\n(\w)")
_SOFT_LINE_BREAK = re.compile(r"(?<![.!?:])\n(?=\S)")


def split_paragraphs(body: str) -> list[str]:
    """Best-effort paragraph splitter for PDF-extracted text.

    Process:
    1. Heal hard-wrapped hyphenated words (``recogni-\\nzed`` → ``recognized``).
    2. Split on blank-line boundaries to get paragraph chunks.
    3. Inside each chunk, collapse remaining single-newline soft breaks
       (visual line wraps from the PDF) into spaces.

    The order matters - if we collapsed soft breaks first the regex
    would treat the second ``\\n`` of a ``\\n\\n`` boundary as a soft
    break and join the two paragraphs together.
    """
    if not body:
        return []
    healed = _HYPHEN_LINE_BREAK.sub(r"\1", body)
    chunks = _PARAGRAPH_BREAK.split(healed)
    out: list[str] = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue
        merged = _SOFT_LINE_BREAK.sub(" ", chunk)
        # Any remaining (sentence-final-followed) hard line break is just
        # a stylistic indent inside the paragraph - turn it into a space.
        merged = merged.replace("\n", " ").strip()
        out.append(" ".join(merged.split()))
    return out


def extract_samples(paper: Paper) -> list[StyleSample]:
    """Return the filtered, ready-to-insert samples for one paper.

    Sections whose titles match the block list (references, appendix,
    acknowledgements, …) are skipped wholesale. Every other section is
    split into paragraphs and each paragraph is fed through
    :func:`is_useful_paragraph`.
    """
    samples: list[StyleSample] = []
    sections = paper.sections or []
    if not sections:
        for paragraph in split_paragraphs(paper.full_text or ""):
            sample = _make_sample(paper.id, "", paragraph)
            if sample is not None:
                samples.append(sample)
        return samples

    for section in sections:
        if section_is_blocked(section.title):
            continue
        for paragraph in split_paragraphs(section.content):
            sample = _make_sample(paper.id, section.title, paragraph)
            if sample is not None:
                samples.append(sample)
    return samples


def extract_samples_from_text(
    text: str,
    *,
    paper_id: str,
    section_title: str = "",
) -> list[StyleSample]:
    """Variant of :func:`extract_samples` that takes raw text.

    Used by S4.1.3 to fold accepted Scribe revisions into the
    fingerprint training set without needing a Paper object.
    Returns an empty list when no paragraph survives the filter.
    """
    samples: list[StyleSample] = []
    for paragraph in split_paragraphs(text or ""):
        sample = _make_sample(paper_id, section_title, paragraph)
        if sample is not None:
            samples.append(sample)
    return samples


def _make_sample(paper_id: str, section_title: str, paragraph: str) -> StyleSample | None:
    if not is_useful_paragraph(paragraph):
        return None
    sentences = split_sentences(paragraph)
    words = paragraph.split()
    return StyleSample(
        id="",
        paper_id=paper_id,
        section_title=section_title,
        paragraph=paragraph,
        char_count=len(paragraph),
        word_count=len(words),
        sentence_count=len(sentences),
    )
