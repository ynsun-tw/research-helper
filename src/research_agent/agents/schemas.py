"""Pydantic v2 schemas for LLM agent JSON replies (Pydantic-AI migration).

Replaces the hand-rolled ``extract_json`` + ``data.get(...)`` pattern that
used to live in every agent. The contract the previous code relied on is
preserved exactly:

* unparsable replies raise :class:`ValueError` (callers already have
  ``except ValueError`` fallbacks; ``ValidationError`` is a subclass so
  this Just Works);
* partial replies → fields default to ``""``/``[]``/``0`` instead of
  crashing;
* out-of-range numbers and mixed-type lists are coerced silently via
  ``mode="before"`` validators (matching the old defensive
  ``_as_str_list`` / clamp helpers);
* extra fields the LLM emits are ignored, not rejected.

The win over the old helpers is that downstream code now gets a typed
object with field-level guarantees (e.g. ``CritiquePayload.support_score``
is *always* a float in [1.0, 9.0]) instead of a bare ``dict[str, Any]``.
"""

from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_TModel = TypeVar("_TModel", bound=BaseModel)


# ----------------------------------------------------------- raw text → JSON

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def strip_to_json(text: str) -> str:
    """Extract the most likely JSON object out of an LLM reply.

    Handles three shapes the wild LLMs reliably produce:

    1. ```` ```json\n{...}\n``` ```` markdown fences.
    2. ``{...}`` surrounded by prose (we slice from first ``{`` to last ``}``).
    3. Bare JSON. We return it as-is.
    """
    stripped = text.strip()
    fence = _FENCE_RE.search(stripped)
    if fence:
        return fence.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_model(text: str, model: type[_TModel]) -> _TModel:
    """Parse a raw LLM reply into ``model``.

    Raises :class:`ValueError` on either JSON-level or schema-level
    failure. Callers that previously caught ``ValueError`` from
    ``extract_json`` keep working with no changes — :class:`ValidationError`
    is itself a ``ValueError`` subclass, and we explicitly catch it for
    nicer error messages anyway.
    """
    payload = strip_to_json(text)
    try:
        return model.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(
            f"{model.__name__} failed schema validation: {exc}"
        ) from exc
    except ValueError as exc:  # JSON decode failure surfaces as ValueError
        raise ValueError(
            f"Could not parse {model.__name__} JSON from model output: {exc}"
        ) from exc


# ------------------------------------------------------ shared coercion shims


def _coerce_str_list(value: Any) -> list[str]:
    """Best-effort cast of an arbitrary value to ``list[str]``.

    Mirrors the old ``_as_str_list`` helper that every agent had a copy
    of: non-lists collapse to ``[]``, falsy elements are dropped, and
    each surviving element is ``str()``-cast and stripped.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        if v in (None, ""):
            continue
        out.append(str(v).strip())
    return [s for s in out if s]


def _clamp_unit(value: Any) -> float:
    """Coerce ``value`` to a float clamped to [0.0, 1.0]; default 0.5."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


def _clamp_unit_zero(value: Any) -> float:
    """Like :func:`_clamp_unit` but defaults to 0.0 on parse failure."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _clean_str(value: Any) -> str:
    """Coerce to a stripped string; ``None`` becomes ``""``."""
    if value is None:
        return ""
    return str(value).strip()


# =========================================================== shared sub-models


class ClaimEvidencePayload(BaseModel):
    """One ``claim`` ↔ ``evidence`` pair.

    Accepts the legacy debate-style aliases ``assumption`` and ``basis``
    that the idea-debate prompts produce, so old prompt YAMLs don't have
    to be touched.
    """

    model_config = ConfigDict(extra="ignore")

    claim: str = ""
    evidence: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if not out.get("claim") and out.get("assumption"):
            out["claim"] = out["assumption"]
        if not out.get("evidence") and out.get("basis"):
            out["evidence"] = out["basis"]
        return out

    @field_validator("claim", "evidence", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)


# =================================================================== Analyst


class AnalysisPayload(BaseModel):
    """Reply shape for :meth:`Analyst.analyze_paper`."""

    model_config = ConfigDict(extra="ignore")

    contributions: list[str] = Field(default_factory=list)
    method_insights: list[str] = Field(default_factory=list)
    potential_impact: str = ""
    related_work: list[str] = Field(default_factory=list)
    claimed_vs_evidence: list[ClaimEvidencePayload] = Field(default_factory=list)
    confidence: float = 0.5

    @field_validator(
        "contributions", "method_insights", "related_work", mode="before"
    )
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("claimed_vs_evidence", mode="before")
    @classmethod
    def _drop_non_dicts(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]

    @field_validator("potential_impact", mode="before")
    @classmethod
    def _stringify_impact(cls, v: Any) -> str:
        return _clean_str(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return _clamp_unit(v)


class IdeaSupportPayload(BaseModel):
    """Reply shape for :meth:`Analyst.analyze_idea`."""

    model_config = ConfigDict(extra="ignore")

    supports: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    evidence: list[ClaimEvidencePayload] = Field(default_factory=list)
    confidence: float = 0.5

    @field_validator("supports", "suggestions", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def _drop_non_dicts(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return _clamp_unit(v)


class ConclusionPayload(BaseModel):
    """Reply shape for follow-up debate turns (Analyst + Critic)."""

    model_config = ConfigDict(extra="ignore")

    conclusion: str = ""

    @field_validator("conclusion", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)


# ==================================================================== Critic


def _normalize_critic_score(value: Any) -> float:
    """Clamp to [1.0, 9.0]; never return 10 (product rule)."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 5.0
    if score >= 10:
        score = 9.0
    return max(1.0, min(9.0, score))


class CritiquePayload(BaseModel):
    """Reply shape for :meth:`Critic.critique_paper` / ``.critique_idea``."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    objections: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    support_score: float = Field(default=5.0, alias="support_score")
    score_reason: str = ""
    honesty_note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_score_aliases(cls, data: Any) -> Any:
        """LLMs occasionally pick ``score`` or ``rating`` over ``support_score``."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "support_score" not in out:
            for alias in ("score", "rating"):
                if alias in out:
                    out["support_score"] = out[alias]
                    break
        return out

    @field_validator("objections", "suggestions", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("support_score", mode="before")
    @classmethod
    def _normalize_score(cls, v: Any) -> float:
        return _normalize_critic_score(v)

    @field_validator("score_reason", "honesty_note", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)


# ===================================================== writing-review pipeline


class WritingReviewPayload(BaseModel):
    """Reply shape for analyst/critic writing reviews (S4.3.1)."""

    model_config = ConfigDict(extra="ignore")

    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("issues", "suggestions", mode="before")
    @classmethod
    def _coerce_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)

    @field_validator("summary", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)


# ==================================================================== Scribe


class DraftPayload(BaseModel):
    """Reply shape for one Scribe draft (S4.2.1) and revision (S4.3.1)."""

    model_config = ConfigDict(extra="ignore")

    draft: str = ""
    style_note: str = ""

    @field_validator("draft", "style_note", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)


# ================================================================== Searcher


class SearcherScoreItem(BaseModel):
    """One ``{index, score, reason}`` entry from ``Searcher.score_hits``."""

    model_config = ConfigDict(extra="ignore")

    index: int | None = None
    score: float | None = None
    reason: str = ""

    @field_validator("index", mode="before")
    @classmethod
    def _coerce_index(cls, v: Any) -> int | None:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, f))

    @field_validator("reason", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)


class SearcherScoresPayload(BaseModel):
    """Reply shape for :meth:`Searcher.score_hits`."""

    model_config = ConfigDict(extra="ignore")

    scores: list[SearcherScoreItem] = Field(default_factory=list)

    @field_validator("scores", mode="before")
    @classmethod
    def _drop_non_dicts(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]


class SearcherRefinementPayload(BaseModel):
    """Reply shape for :meth:`Searcher.suggest_refinement`."""

    model_config = ConfigDict(extra="ignore")

    query: str = ""
    mode: str | None = None
    reason: str = ""
    confidence: float = 0.0

    @field_validator("query", "reason", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v: Any) -> str | None:
        if v in (None, ""):
            return None
        return str(v).strip() or None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return _clamp_unit_zero(v)


# ================================================================ Illustrator


class FigurePayload(BaseModel):
    """Reply shape for one :class:`Illustrator` variant (E5.3).

    Optional fields cover the concept-art variant (``target_model`` and
    ``negative_prompt`` are only meaningful for text-to-image prompts).
    """

    model_config = ConfigDict(extra="ignore")

    code: str = ""
    style_label: str = ""
    notes: str = ""
    suggested_use: str = ""
    target_model: str = ""
    negative_prompt: str = ""

    @field_validator(
        "code",
        "style_label",
        "notes",
        "suggested_use",
        "target_model",
        "negative_prompt",
        mode="before",
    )
    @classmethod
    def _stringify(cls, v: Any) -> str:
        return _clean_str(v)
