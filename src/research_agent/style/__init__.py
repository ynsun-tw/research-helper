"""M4 — style training & writing assistant.

This package powers:

- ``research style train`` (S4.1.1): turn user-supplied papers (PDFs
  on disk, arXiv ids, …) into a corpus of prose paragraphs we can
  later inspect to learn the user's voice.
- ``research style fingerprint`` (S4.1.2): roll those paragraphs up
  into a macro + micro + markers ``Fingerprint`` JSON living under
  ``~/.research-agent/style/``.
- ``research write <section>`` (S4.2.1+): Scribe agent that emits
  drafts grounded in that fingerprint.
- Self-plagiarism detection (S4.4): TF-IDF against the same sample
  corpus so the Scribe never copies the user's earlier wording too
  closely.

The package is intentionally framework-light - just dataclasses, a
SQLite repository, and a few text heuristics. Heavy lifting (LLM
analysis, embedding queries) lives in ``agents/scribe.py`` and
``agents/style_analyzer.py`` (added in S4.1.2).
"""

from research_agent.style.analyzer import StyleAnalyzer
from research_agent.style.extractor import extract_samples, extract_samples_from_text
from research_agent.style.filters import is_useful_paragraph
from research_agent.style.fingerprint import (
    Fingerprint,
    MacroFingerprint,
    MicroFingerprint,
    PersonalMarkers,
)
from research_agent.style.samples import StyleSample, StyleSampleRepository

__all__ = [
    "Fingerprint",
    "MacroFingerprint",
    "MicroFingerprint",
    "PersonalMarkers",
    "StyleAnalyzer",
    "StyleSample",
    "StyleSampleRepository",
    "extract_samples",
    "extract_samples_from_text",
    "is_useful_paragraph",
]
