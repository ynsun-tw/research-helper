"""Paragraph-level self-plagiarism detection via TF-IDF + cosine.

M4 S4.4.1 wants a cheap, deterministic guard against a user
unintentionally reusing too much of their own published wording in a
new draft. The detector compares each paragraph of the input draft
against every paragraph in the ``style_samples`` corpus and warns
when paragraph-level cosine similarity crosses a configurable
threshold (default 0.4 per the milestone spec).

Implementation is intentionally dependency-free: pure-Python TF-IDF
over word unigrams. The corpus sizes we deal with (hundreds of
paragraphs, not millions) make this comfortable, and avoiding scipy
/ sklearn keeps install footprint small.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

DEFAULT_THRESHOLD = 0.40
MAX_SUGGESTIONS_PER_MATCH = 2

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")


def tokenize(text: str) -> list[str]:
    """Cheap word tokenizer.

    Lowercases everything; keeps internal apostrophes / hyphens (so
    ``state-of-the-art`` and ``don't`` survive); drops short tokens
    (<=2 chars) and numeric tokens to reduce noise from page numbers,
    citation indices, and one-letter variables.
    """
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if len(t) > 2]


@dataclass(slots=True)
class ParagraphMatch:
    """One draft paragraph that resembles one corpus paragraph."""

    draft_index: int
    draft_paragraph: str
    source_id: str
    source_paragraph: str
    similarity: float


@dataclass(slots=True)
class SimilarityReport:
    """Aggregated detector output."""

    threshold: float
    draft_paragraph_count: int
    matches: list[ParagraphMatch] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.matches

    @property
    def above_threshold_count(self) -> int:
        return sum(1 for m in self.matches if m.similarity >= self.threshold)

    def to_markdown(self) -> str:
        if self.is_clean:
            return (
                "# Self-plagiarism check\n\n"
                f"No matches found above {self.threshold:.0%} similarity "
                f"across {self.draft_paragraph_count} draft paragraphs."
            )
        lines = [
            "# Self-plagiarism check",
            "",
            f"**{len(self.matches)} match(es)** at or above "
            f"{self.threshold:.0%} similarity "
            f"(checked {self.draft_paragraph_count} draft paragraphs).",
            "",
        ]
        for i, m in enumerate(self.matches, 1):
            lines.append(
                f"## Match {i} — {m.similarity:.0%} similarity "
                f"(draft paragraph {m.draft_index + 1} ↔ `{m.source_id}`)"
            )
            lines.append("")
            lines.append("**Draft paragraph:**")
            lines.append("")
            lines.append("> " + _quote(m.draft_paragraph))
            lines.append("")
            lines.append("**Matching source paragraph:**")
            lines.append("")
            lines.append("> " + _quote(m.source_paragraph))
            lines.append("")
            for s in suggest_rewrites(m):
                lines.append(f"- {s}")
            lines.append("")
        return "\n".join(lines)


def suggest_rewrites(match: ParagraphMatch) -> list[str]:
    """Heuristic rewrite suggestions for a flagged match.

    The point isn't to be clever - just to give the user concrete
    levers to pull instead of a bare percentage. Suggestions scale
    with similarity: higher overlap → more aggressive rewrite advice.
    """
    suggestions: list[str] = []
    if match.similarity >= 0.7:
        suggestions.append(
            "Rewrite this paragraph from scratch - the wording is too close "
            "to your earlier paper to be safe."
        )
        suggestions.append(
            "Switch the framing: lead with the new contribution and cite the "
            "earlier paper for the background instead of repeating it."
        )
    elif match.similarity >= 0.5:
        suggestions.append(
            "Paraphrase the overlapping sentences. Vary sentence order and "
            "swap key phrases for synonyms."
        )
        suggestions.append(
            "Cite the earlier work explicitly so the reused framing is "
            "attributed."
        )
    else:
        suggestions.append(
            "Trim or merge the overlapping sentences to compress the shared "
            "framing into a single, briefer reference."
        )
        suggestions.append(
            "Cite the earlier paper when reusing this background."
        )
    return suggestions[:MAX_SUGGESTIONS_PER_MATCH]


class PlagiarismDetector:
    """Paragraph-level TF-IDF + cosine self-plagiarism check."""

    def __init__(self, *, threshold: float = DEFAULT_THRESHOLD) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold!r}")
        self.threshold = threshold

    def check(
        self,
        draft_paragraphs: list[str],
        corpus_paragraphs: list[tuple[str, str]],
    ) -> SimilarityReport:
        """Return a :class:`SimilarityReport`.

        ``draft_paragraphs`` is the list of paragraphs already split
        out of the new draft. ``corpus_paragraphs`` is a list of
        ``(source_id, paragraph_text)`` pairs from the user's
        existing ``style_samples`` corpus.

        For each draft paragraph we take the *best* corpus match,
        then keep it only if it crosses ``self.threshold``. We never
        emit more than one match per draft paragraph.
        """
        if not draft_paragraphs:
            return SimilarityReport(threshold=self.threshold, draft_paragraph_count=0)
        if not corpus_paragraphs:
            return SimilarityReport(
                threshold=self.threshold,
                draft_paragraph_count=len(draft_paragraphs),
            )

        all_docs = [tokenize(p) for p in draft_paragraphs] + [
            tokenize(p) for _, p in corpus_paragraphs
        ]
        tfidf = _tfidf_vectors(all_docs)
        draft_vecs = tfidf[: len(draft_paragraphs)]
        corpus_vecs = tfidf[len(draft_paragraphs) :]

        matches: list[ParagraphMatch] = []
        for i, dv in enumerate(draft_vecs):
            best_score = 0.0
            best_idx = -1
            for j, cv in enumerate(corpus_vecs):
                sim = _cosine(dv, cv)
                if sim > best_score:
                    best_score = sim
                    best_idx = j
            if best_idx >= 0 and best_score >= self.threshold:
                source_id, source_text = corpus_paragraphs[best_idx]
                matches.append(
                    ParagraphMatch(
                        draft_index=i,
                        draft_paragraph=draft_paragraphs[i],
                        source_id=source_id,
                        source_paragraph=source_text,
                        similarity=best_score,
                    )
                )
        # Sort by similarity descending so the worst offender is on top.
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return SimilarityReport(
            threshold=self.threshold,
            draft_paragraph_count=len(draft_paragraphs),
            matches=matches,
        )


# --- internals --------------------------------------------------------------


def _tfidf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    """Pure-Python TF-IDF (sublinear TF, smoothed IDF).

    Returns one dict per document mapping term → weight. We do NOT
    L2-normalise here - the cosine helper normalises on the fly so
    norm computation can be cached if needed.
    """
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1

    out: list[dict[str, float]] = []
    for d in docs:
        if not d:
            out.append({})
            continue
        tf: Counter[str] = Counter(d)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            # Sublinear TF + smoothed IDF (à la sklearn).
            tf_w = 1.0 + math.log(count)
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            vec[term] = tf_w * idf
        out.append(vec)
    return out


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    # Iterate over the smaller dict for the dot product
    small, large = (v1, v2) if len(v1) < len(v2) else (v2, v1)
    dot = 0.0
    for t, w in small.items():
        if t in large:
            dot += w * large[t]
    if dot == 0.0:
        return 0.0
    n1 = math.sqrt(sum(w * w for w in v1.values()))
    n2 = math.sqrt(sum(w * w for w in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def _quote(text: str, *, max_chars: int = 600) -> str:
    """Newline-collapsed paragraph quote, truncated to keep reports compact."""
    flattened = " ".join((text or "").split())
    if len(flattened) > max_chars:
        return flattened[:max_chars] + " …"
    return flattened


def split_into_paragraphs(text: str) -> list[str]:
    """Cheap blank-line paragraph splitter for the draft side.

    The corpus side already comes pre-split from the ``style_samples``
    table; the draft is whatever the user passes in, so we normalise
    it here.
    """
    if not text:
        return []
    chunks = re.split(r"\n\s*\n+", text)
    return [c.strip() for c in chunks if c and c.strip()]


def from_iterable(samples: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Convenience wrapper: materialise an iterable of (id, paragraph) pairs."""
    return list(samples)
