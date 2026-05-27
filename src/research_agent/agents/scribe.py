"""Scribe agent — multi-version section drafting in the user's voice (E4.2).

Given a section type ("abstract" / "introduction" / "related_work" / ...)
and a :class:`Fingerprint`, the Scribe emits N drafts that each lean
into a different slice of the user's style (concise, technical depth,
narrative arc). Drafts are produced via parallel LLM calls so the
user sees the full bouquet in roughly the same wall-clock time as a
single call.

Self-plagiarism guard, context-aware writing (--context), and the
auto-review pipeline land in subsequent stories (S4.2.2 / S4.3.x /
S4.4.1) and consume the same :class:`Draft` envelope this module
defines.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent
from research_agent.agents.schemas import DraftPayload, parse_model
from research_agent.agents.writing_pipeline import (
    WritingReview,
    build_revision_prompt,
)
from research_agent.style.fingerprint import Fingerprint

# Canonical section names that the Scribe knows about.
SECTION_TYPES: tuple[str, ...] = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "results",
    "discussion",
    "conclusion",
)

_SECTION_ALIASES: dict[str, str] = {
    "intro": "introduction",
    "related": "related_work",
    "relatedwork": "related_work",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "experiments": "results",
    "evaluation": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusions": "conclusion",
}


def normalize_section(name: str) -> str:
    """Map free-form section names to the canonical set.

    Raises :class:`ValueError` on truly unknown sections so the CLI can
    surface a helpful error instead of silently calling the LLM with
    garbage.
    """
    if not name:
        raise ValueError("section name is required")
    lowered = name.strip().lower().replace("-", "_").replace(" ", "_")
    if lowered in SECTION_TYPES:
        return lowered
    if lowered in _SECTION_ALIASES:
        return _SECTION_ALIASES[lowered]
    raise ValueError(
        f"Unknown section: {name!r}. Valid: {', '.join(SECTION_TYPES)}."
    )


# (label, directive) pairs used in round-robin order when the caller
# asks for N drafts. The directive is dropped verbatim into the user
# prompt so it must be understandable to the LLM, not just to us.
DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    (
        "concise",
        "Lean toward tighter sentences. Cut hedges. Lead with the punchline.",
    ),
    (
        "technical depth",
        "Lean toward precise terminology and slightly longer compound sentences. "
        "Use fewer transitions; let the technical structure do the work.",
    ),
    (
        "narrative arc",
        "Lean toward an explicit motivational arc: situate the problem, build "
        "tension, then resolve. Use more transitions and contextual framing.",
    ),
)


@dataclass
class Draft:
    """One Scribe output. Stable shape across S4.2.x / S4.3.x."""

    section: str
    version: str  # "A" | "B" | "C" | ...
    variant_label: str
    text: str
    style_note: str
    word_count: int = 0
    target_words: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class Scribe(BaseAgent):
    """Section-draft generator. Voice-matched, multi-version."""

    role = "scribe"

    @property
    def prompt_name(self) -> str:
        return "scribe"

    def run(self, context: dict[str, Any]) -> AgentResponse:
        """Adapter used by :class:`Orchestrator`. Returns the first draft."""
        section = context.get("section", "")
        fingerprint = context.get("fingerprint")
        if not isinstance(fingerprint, Fingerprint) and fingerprint is not None:
            raise TypeError("context['fingerprint'] must be a Fingerprint instance")
        drafts = self.generate(
            normalize_section(str(section)),
            fingerprint=fingerprint,
            context=str(context.get("user_context", "")),
            target_words=int(context.get("target_words", 300)),
            n=int(context.get("n", 1)),
            parallel=bool(context.get("parallel", True)),
        )
        primary = drafts[0]
        return AgentResponse(
            role="scribe",
            content=primary.text,
            metadata={"drafts": drafts},
        )

    def generate(
        self,
        section: str,
        *,
        fingerprint: Fingerprint | None = None,
        context: str = "",
        target_words: int = 300,
        n: int = 3,
        parallel: bool = True,
    ) -> list[Draft]:
        """Produce ``n`` drafts of ``section`` in the user's voice.

        Variants cycle through :data:`DEFAULT_VARIANTS` (concise →
        technical → narrative → concise → …). Each variant becomes one
        LLM call; calls run on a thread pool when ``parallel`` is true.

        ``fingerprint`` may be ``None`` for users who haven't trained
        yet - the Scribe will fall back to generic academic prose,
        and the resulting drafts make that explicit in their
        :attr:`Draft.style_note`.
        """
        canonical = normalize_section(section)
        if n <= 0:
            return []
        variants = [DEFAULT_VARIANTS[i % len(DEFAULT_VARIANTS)] for i in range(n)]
        versions = [chr(ord("A") + i) for i in range(n)]
        prompts = [
            _build_user_prompt(
                section=canonical,
                fingerprint=fingerprint,
                context=context,
                target_words=target_words,
                variant_label=label,
                variant_directive=directive,
            )
            for label, directive in variants
        ]

        if parallel and n > 1:
            with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
                raw_outputs = list(pool.map(self._chat, prompts))
        else:
            raw_outputs = [self._chat(p) for p in prompts]

        drafts: list[Draft] = []
        # 3.9 lacks ``zip(strict=True)``; assert the invariant explicitly so
        # mismatched lengths still surface loudly instead of silently
        # truncating the draft set.
        assert len(versions) == len(variants) == len(raw_outputs), (
            "scribe internal: versions/variants/raw_outputs length mismatch"
        )
        for version, (label, _directive), raw in zip(versions, variants, raw_outputs):
            text, note = _parse_draft(raw)
            drafts.append(
                Draft(
                    section=canonical,
                    version=version,
                    variant_label=label,
                    text=text,
                    style_note=note,
                    word_count=len(text.split()),
                    target_words=target_words,
                )
            )
        return drafts

    def revise(
        self,
        draft: Draft,
        reviews: list[WritingReview],
        *,
        fingerprint: Fingerprint | None = None,
    ) -> Draft:
        """Produce a revised draft that addresses ``reviews``.

        Used by the auto-review pipeline (S4.3.1). The revision keeps
        the same section / target word count / version label as the
        original; the resulting ``style_note`` records what changed
        instead of the original voicing label.
        """
        # Reviews with nothing actionable shortcut to a no-op revision:
        # we still return a Draft so downstream code can treat the path
        # uniformly, but we don't waste an LLM call.
        if not any(r.issues or r.suggestions or r.summary for r in reviews):
            return Draft(
                section=draft.section,
                version=draft.version,
                variant_label=draft.variant_label,
                text=draft.text,
                style_note="No actionable review feedback - draft unchanged.",
                word_count=draft.word_count,
                target_words=draft.target_words,
            )

        prompt = build_revision_prompt(draft=draft, reviews=reviews)
        # Prepend a compact fingerprint reminder so the revised draft
        # doesn't drift in voice while addressing the feedback.
        if fingerprint is not None and fingerprint.sample_count > 0:
            prompt = (
                "Style fingerprint to keep matching:\n"
                + _format_fingerprint(fingerprint)
                + "\n\n"
                + prompt
            )
        raw = self._chat(prompt, temperature=0.5)
        text, note = _parse_draft(raw)
        return Draft(
            section=draft.section,
            version=draft.version,
            variant_label=draft.variant_label,
            text=text,
            style_note=note or "Revised in response to reviewer feedback.",
            word_count=len(text.split()),
            target_words=draft.target_words,
        )


# ---------------------------------------------------------------- prompt build

def _build_user_prompt(
    *,
    section: str,
    fingerprint: Fingerprint | None,
    context: str,
    target_words: int,
    variant_label: str,
    variant_directive: str,
) -> str:
    lines = [
        f"Section to draft: **{section}**",
        f"Target length: ~{target_words} words (±20%).",
        f"Variant directive ({variant_label}): {variant_directive}",
    ]
    if context.strip():
        lines.extend(["", "User-supplied research context:", context.strip()])
    if fingerprint is not None and fingerprint.sample_count > 0:
        lines.extend(["", "Style fingerprint to mimic:", _format_fingerprint(fingerprint)])
    else:
        lines.extend(
            [
                "",
                "No style fingerprint available yet - write in clean, neutral "
                "academic English and note that fact in your style_note.",
            ]
        )
    lines.extend(
        [
            "",
            "Return JSON only: {\"draft\": \"...\", \"style_note\": \"...\"}.",
        ]
    )
    return "\n".join(lines)


def _format_fingerprint(fp: Fingerprint) -> str:
    """A compact human / LLM-readable rendering of the fingerprint.

    Skips zero-valued micro fields so the prompt isn't padded with
    noise that dilutes the meaningful signal.
    """
    parts: list[str] = []
    macro = fp.macro
    if macro.abstract_opener:
        parts.append(f"- abstract opener: \"{macro.abstract_opener} ...\"")
    if macro.intro_opener:
        parts.append(f"- intro opener: \"{macro.intro_opener} ...\"")
    if macro.related_work_strategy:
        parts.append(f"- related-work strategy: {macro.related_work_strategy}")
    if macro.intro_avg_paragraphs > 0:
        parts.append(f"- avg paragraphs per intro: {macro.intro_avg_paragraphs:.1f}")

    micro = fp.micro
    if micro.avg_sentence_length > 0:
        parts.append(
            f"- sentence length (words): mean {micro.avg_sentence_length:.1f}, "
            f"median {micro.median_sentence_length:.1f}, "
            f"p10/p90 {micro.p10_sentence_length:.0f}/{micro.p90_sentence_length:.0f}"
        )
    if micro.avg_paragraph_length > 0:
        parts.append(
            f"- paragraph length: ~{micro.avg_paragraph_length:.1f} sentences"
        )
    top_transitions = sorted(
        micro.transition_freq.items(), key=lambda kv: kv[1], reverse=True
    )[:5]
    if top_transitions:
        parts.append(
            "- transition preferences (per-100-sentences): "
            + ", ".join(f"{w} {v:.1f}" for w, v in top_transitions)
        )
    if micro.hedging_per_100 or micro.confidence_per_100:
        parts.append(
            f"- voice rates per 100 sentences: hedging {micro.hedging_per_100:.1f}, "
            f"confidence {micro.confidence_per_100:.1f}, passive {micro.passive_per_100:.1f}"
        )

    markers = fp.markers
    if markers.citation_format:
        parts.append(f"- citation format: {markers.citation_format}")
    if markers.figure_ref_format:
        parts.append(f"- figure references: {markers.figure_ref_format}")
    if markers.table_ref_format:
        parts.append(f"- table references: {markers.table_ref_format}")
    if markers.em_dash_per_100 > 0:
        parts.append(f"- em-dash rate: {markers.em_dash_per_100:.1f} per 100 sentences")

    if not parts:
        return "- (no actionable signal in the fingerprint yet)"
    return "\n".join(parts)


def _parse_draft(raw: str) -> tuple[str, str]:
    """Pull (draft_text, style_note) out of one Scribe LLM reply.

    Tolerates: extra prose around the JSON, missing fields, and the LLM
    forgetting to emit JSON at all (we treat the whole reply as the
    draft in that fallback case).
    """
    try:
        payload = parse_model(raw, DraftPayload)
    except ValueError:
        return raw.strip(), ""
    draft = payload.draft or raw.strip()
    return draft, payload.style_note
