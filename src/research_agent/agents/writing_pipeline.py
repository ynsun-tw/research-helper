"""Auto-review pipeline for Scribe drafts (M4 S4.3.1).

Glue between the Scribe, the writing-focused Analyst, and the
writing-focused Critic:

1. Scribe produces a draft (S4.2.1 / S4.2.2).
2. Analyst + Critic review it in parallel (separate prompts from
   their paper-analysis variants - see ``analyst_writing.yaml`` and
   ``critic_writing.yaml``).
3. Scribe revises the draft based on the combined feedback.

Shapes here (``WritingReview`` and ``ReviewedDraft``) are the API
contract for downstream S4.3.2 (user picks which suggestions to
accept) and S4.1.3 (track the (original, revised) pair as a
fingerprint update signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from research_agent.agents.base import extract_json

if TYPE_CHECKING:
    from research_agent.agents.scribe import Draft

# Roles that show up inside :class:`WritingReview`. Stable strings so
# downstream code can do simple equality checks.
REVIEW_ROLES = ("analyst", "critic")


@dataclass(slots=True)
class WritingReview:
    """One reviewer's verdict on a Scribe draft."""

    role: str
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""


@dataclass(slots=True)
class ReviewedDraft:
    """Full output of the writing-review pipeline."""

    original: Draft
    reviews: list[WritingReview]
    revised: Draft

    @property
    def analyst_review(self) -> WritingReview | None:
        for r in self.reviews:
            if r.role == "analyst":
                return r
        return None

    @property
    def critic_review(self) -> WritingReview | None:
        for r in self.reviews:
            if r.role == "critic":
                return r
        return None

    def all_issues(self) -> list[str]:
        out: list[str] = []
        for r in self.reviews:
            out.extend(r.issues)
        return out


def parse_review(role: str, raw: str) -> WritingReview:
    """Parse a writing-review LLM reply into a :class:`WritingReview`.

    Tolerates: missing fields, missing JSON, lists with non-string
    items, extra prose around the JSON envelope.
    """
    try:
        data = extract_json(raw)
    except ValueError:
        return WritingReview(role=role, summary=raw.strip()[:280], raw_response=raw)
    issues = _as_str_list(data.get("issues"))
    suggestions = _as_str_list(data.get("suggestions"))
    summary = str(data.get("summary", "")).strip()
    return WritingReview(
        role=role,
        issues=issues,
        suggestions=suggestions,
        summary=summary,
        raw_response=raw,
    )


def build_review_prompt(*, draft_text: str, section: str) -> str:
    """User-side prompt for an Analyst/Critic writing review."""
    return (
        f"Section under review: **{section}**\n\n"
        f"Draft text:\n{draft_text}\n\n"
        "Return JSON only: {\"issues\": [...], \"suggestions\": [...], \"summary\": \"...\"}."
    )


def build_revision_prompt(*, draft: Draft, reviews: list[WritingReview]) -> str:
    """User-side prompt asking the Scribe to revise based on reviews."""
    lines = [
        f"You earlier produced this draft of the **{draft.section}** section. "
        "Two reviewers have flagged issues. Produce a single revised draft "
        "that addresses the issues while preserving the user's voice and "
        "keeping the draft length within ±20% of the original target.",
        "",
        "[ORIGINAL DRAFT]",
        draft.text,
        "",
    ]
    for r in reviews:
        if not (r.issues or r.suggestions or r.summary):
            continue
        lines.append(f"[{r.role.upper()} REVIEW]")
        if r.summary:
            lines.append(f"Summary: {r.summary}")
        if r.issues:
            lines.append("Issues:")
            lines.extend(f"- {x}" for x in r.issues)
        if r.suggestions:
            lines.append("Suggestions:")
            lines.extend(f"- {x}" for x in r.suggestions)
        lines.append("")
    lines.append(
        "Return JSON only: "
        "{\"draft\": \"the revised section text\", "
        "\"style_note\": \"what you changed and why\"}."
    )
    return "\n".join(lines)


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v]
