"""Compute a :class:`Fingerprint` from a corpus of :class:`StyleSample` rows.

Everything here is deterministic and offline - no LLM calls. That
keeps unit tests cheap, makes the fingerprint reproducible across
machines, and means style training doesn't burn tokens. An LLM
augmentation hook (the :attr:`MacroFingerprint.notes` field) is
reserved for a future story.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable

from research_agent.style.filters import split_sentences
from research_agent.style.fingerprint import (
    Fingerprint,
    MacroFingerprint,
    MicroFingerprint,
    PersonalMarkers,
)
from research_agent.style.samples import StyleSample

# --- Word lists (kept tight on purpose - quality > coverage) ---------------
_TRANSITION_WORDS: tuple[str, ...] = (
    "however",
    "moreover",
    "furthermore",
    "therefore",
    "thus",
    "hence",
    "nevertheless",
    "consequently",
    "additionally",
    "instead",
    "indeed",
    "whereas",
    "while",
    "although",
    "specifically",
)
_HEDGING_WORDS: tuple[str, ...] = (
    "may",
    "might",
    "could",
    "perhaps",
    "possibly",
    "potentially",
    "likely",
    "suggest",
    "suggests",
    "appear",
    "appears",
    "seems",
    "seem",
    "presumably",
)
_CONFIDENCE_WORDS: tuple[str, ...] = (
    "demonstrate",
    "demonstrates",
    "prove",
    "proves",
    "proven",
    "clearly",
    "evidently",
    "definitively",
    "definitely",
    "necessarily",
    "must",
    "establish",
    "establishes",
)

_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", re.IGNORECASE
)
_CITATION_BRACKET_RE = re.compile(r"\[\d+(?:\s*[,\-]\s*\d+)*\]")
_CITATION_AUTHOR_YEAR_RE = re.compile(
    r"\(\s*[A-Z][A-Za-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][A-Za-z]+))?,\s*\d{4}[a-z]?\s*\)"
)
_CITATION_LATEX_RE = re.compile(r"\\cite[a-zA-Z]*\{[^}]+\}")
_FIGURE_FULL_RE = re.compile(r"\bFigure\s+\d+", re.IGNORECASE)
_FIGURE_ABBR_RE = re.compile(r"\bFig\.\s*\d+", re.IGNORECASE)
_TABLE_FULL_RE = re.compile(r"\bTable\s+\d+", re.IGNORECASE)
_TABLE_ABBR_RE = re.compile(r"\bTab\.\s*\d+", re.IGNORECASE)


def _word_re(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


_TRANSITION_REGEXES = [(w, _word_re(w)) for w in _TRANSITION_WORDS]
_HEDGING_REGEXES = [_word_re(w) for w in _HEDGING_WORDS]
_CONFIDENCE_REGEXES = [_word_re(w) for w in _CONFIDENCE_WORDS]


def _classify_section(title: str) -> str:
    """Normalize ``section_title`` strings to a small canonical set."""
    lower = (title or "").lower()
    if "abstract" in lower:
        return "abstract"
    if "introduction" in lower or " intro" in lower or lower.startswith("intro"):
        return "intro"
    if "related" in lower or "background" in lower or "prior" in lower:
        return "related"
    return "other"


def _opener_signature(paragraph: str, *, words: int = 5) -> str:
    """Return the lowercased first ``words`` words - used to bucket paragraphs
    by opening pattern."""
    sentences = split_sentences(paragraph)
    if not sentences:
        return ""
    first = sentences[0].split()
    return " ".join(first[:words]).lower()


def _most_common_opener(paragraphs: Iterable[str]) -> str:
    sigs = [_opener_signature(p) for p in paragraphs]
    sigs = [s for s in sigs if s]
    if not sigs:
        return ""
    return Counter(sigs).most_common(1)[0][0]


def _avg(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    # statistics.quantiles needs n>=2 and returns n-1 cut points; we just
    # do a manual linear interpolation - good enough for sample sizes
    # in the hundreds.
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    if f == c:
        return float(sorted_v[f])
    return float(sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f))


class StyleAnalyzer:
    """Turn a list of :class:`StyleSample` rows into a :class:`Fingerprint`."""

    def analyze(self, samples: list[StyleSample]) -> Fingerprint:
        if not samples:
            return Fingerprint()
        macro = self.analyze_macro(samples)
        micro = self.analyze_micro(samples)
        markers = self.extract_markers(samples)
        paper_ids = {s.paper_id for s in samples}
        return Fingerprint(
            macro=macro,
            micro=micro,
            markers=markers,
            sample_count=len(samples),
            paper_count=len(paper_ids),
            version=1,
        )

    # ------------------------------------------------------------------ macro
    def analyze_macro(self, samples: list[StyleSample]) -> MacroFingerprint:
        by_section: dict[str, list[StyleSample]] = {
            "abstract": [],
            "intro": [],
            "related": [],
            "other": [],
        }
        for s in samples:
            by_section[_classify_section(s.section_title)].append(s)

        abstract_paragraphs = [s.paragraph for s in by_section["abstract"]]
        intro_paragraphs = [s.paragraph for s in by_section["intro"]]
        related_paragraphs = [s.paragraph for s in by_section["related"]]

        # Per-paper averages for paragraph counts.
        intro_per_paper: dict[str, int] = Counter(s.paper_id for s in by_section["intro"])
        intro_avg_paragraphs = _avg([float(v) for v in intro_per_paper.values()])

        # Section count = distinct section_title per paper, averaged.
        sections_per_paper: dict[str, set[str]] = {}
        for s in samples:
            sections_per_paper.setdefault(s.paper_id, set()).add(s.section_title)
        section_count_avg = _avg([float(len(v)) for v in sections_per_paper.values()])

        return MacroFingerprint(
            abstract_opener=_most_common_opener(abstract_paragraphs),
            abstract_avg_sentences=_avg(
                [float(s.sentence_count) for s in by_section["abstract"]]
            ),
            intro_opener=_most_common_opener(intro_paragraphs),
            intro_avg_paragraphs=intro_avg_paragraphs,
            related_work_strategy=_related_work_strategy(related_paragraphs),
            section_count_avg=section_count_avg,
            notes="",
        )

    # ------------------------------------------------------------------ micro
    def analyze_micro(self, samples: list[StyleSample]) -> MicroFingerprint:
        sentence_lengths: list[float] = []
        paragraph_lengths: list[float] = []
        transition_counts: Counter[str] = Counter()
        hedging_total = 0
        confidence_total = 0
        passive_total = 0
        tokens: list[str] = []

        for s in samples:
            sentences = split_sentences(s.paragraph)
            if not sentences:
                continue
            paragraph_lengths.append(float(len(sentences)))
            for sent in sentences:
                words = sent.split()
                sentence_lengths.append(float(len(words)))
                tokens.extend(w.lower() for w in words)
            for word, regex in _TRANSITION_REGEXES:
                transition_counts[word] += len(regex.findall(s.paragraph))
            hedging_total += sum(len(r.findall(s.paragraph)) for r in _HEDGING_REGEXES)
            confidence_total += sum(
                len(r.findall(s.paragraph)) for r in _CONFIDENCE_REGEXES
            )
            passive_total += len(_PASSIVE_RE.findall(s.paragraph))

        n_sent = len(sentence_lengths)

        def per_100(count: int) -> float:
            return (count * 100.0 / n_sent) if n_sent else 0.0

        transition_freq = {
            word: per_100(count) for word, count in transition_counts.items() if count
        }

        ttr = (len(set(tokens)) / len(tokens)) if tokens else 0.0

        return MicroFingerprint(
            avg_sentence_length=_avg(sentence_lengths),
            median_sentence_length=(
                float(statistics.median(sentence_lengths)) if sentence_lengths else 0.0
            ),
            p10_sentence_length=_percentile(sentence_lengths, 0.10),
            p90_sentence_length=_percentile(sentence_lengths, 0.90),
            avg_paragraph_length=_avg(paragraph_lengths),
            sentence_count=n_sent,
            transition_freq=transition_freq,
            hedging_per_100=per_100(hedging_total),
            confidence_per_100=per_100(confidence_total),
            passive_per_100=per_100(passive_total),
            type_token_ratio=float(ttr),
        )

    # ---------------------------------------------------------------- markers
    def extract_markers(self, samples: list[StyleSample]) -> PersonalMarkers:
        title_counter: Counter[str] = Counter()
        for s in samples:
            if s.section_title:
                title_counter[s.section_title] += 1
        top_titles = [t for t, _ in title_counter.most_common(10)]

        latex_hits = 0
        bracket_hits = 0
        author_hits = 0
        figure_full = 0
        figure_abbr = 0
        table_full = 0
        table_abbr = 0
        em_dashes = 0
        n_sent = 0
        for s in samples:
            latex_hits += len(_CITATION_LATEX_RE.findall(s.paragraph))
            bracket_hits += len(_CITATION_BRACKET_RE.findall(s.paragraph))
            author_hits += len(_CITATION_AUTHOR_YEAR_RE.findall(s.paragraph))
            figure_full += len(_FIGURE_FULL_RE.findall(s.paragraph))
            figure_abbr += len(_FIGURE_ABBR_RE.findall(s.paragraph))
            table_full += len(_TABLE_FULL_RE.findall(s.paragraph))
            table_abbr += len(_TABLE_ABBR_RE.findall(s.paragraph))
            em_dashes += s.paragraph.count("—")
            n_sent += s.sentence_count

        citation_format = _pick_format(
            {
                "latex_cite": latex_hits,
                "bracket_num": bracket_hits,
                "author_year": author_hits,
            }
        )
        figure_ref_format = _pick_two(figure_full, figure_abbr, "Figure", "Fig.")
        table_ref_format = _pick_two(table_full, table_abbr, "Table", "Tab.")
        em_dash_per_100 = (em_dashes * 100.0 / n_sent) if n_sent else 0.0

        return PersonalMarkers(
            top_section_titles=top_titles,
            citation_format=citation_format,
            figure_ref_format=figure_ref_format,
            table_ref_format=table_ref_format,
            em_dash_per_100=em_dash_per_100,
        )


def _related_work_strategy(paragraphs: list[str]) -> str:
    """Cheap heuristic for how the author organizes related-work prose.

    - If years appear densely → ``chronological``.
    - If theme-marker words (``approach``, ``method``, ``family``,
      ``stream``) dominate → ``thematic``.
    - If contrast markers (``unlike``, ``in contrast``, ``differ``)
      dominate → ``comparison``.
    - Otherwise empty.
    """
    if not paragraphs:
        return ""
    blob = " ".join(paragraphs).lower()
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", blob))
    thematic_hits = sum(
        blob.count(w) for w in ("approach", "method", "family", "line of work")
    )
    contrast_hits = sum(blob.count(w) for w in ("unlike", "in contrast", "differ"))
    scores = {
        "chronological": year_hits,
        "thematic": thematic_hits,
        "comparison": contrast_hits,
    }
    winner, score = max(scores.items(), key=lambda kv: kv[1])
    return winner if score >= 2 else ""


def _pick_format(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return ""
    top, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count / total >= 0.7:
        return top
    # No format dominates -> "mixed", unless everything is tiny noise.
    if total < 3:
        return ""
    return "mixed"


def _pick_two(a_count: int, b_count: int, a_label: str, b_label: str) -> str:
    if a_count == 0 and b_count == 0:
        return ""
    return a_label if a_count >= b_count else b_label
