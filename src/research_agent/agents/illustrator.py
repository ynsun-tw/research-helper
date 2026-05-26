"""Illustrator agent — figure code/prompt generator (E5.3).

Given a figure type and a short description, the Illustrator emits N
variant drafts:

* ``architecture`` → TikZ snippets, paste straight into LaTeX.
* ``result`` → matplotlib/seaborn Python scripts, runnable as-is.
* ``concept`` → text-to-image prompts for DALL·E 3 / Midjourney / SD.

Each figure type owns its own system prompt (``illustrator_*.yaml``)
so the LLM is told upfront which output schema to honor. Variants
cycle through type-specific layout/chart/viewpoint directives so a
``--versions 3`` call returns three meaningfully different drafts in
parallel, the same multi-version pattern as :class:`Scribe`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent
from research_agent.agents.schemas import FigurePayload, parse_model

FIGURE_TYPES: tuple[str, ...] = ("architecture", "result", "concept")

_FIGURE_TYPE_ALIASES: dict[str, str] = {
    "arch": "architecture",
    "system": "architecture",
    "pipeline": "architecture",
    "diagram": "architecture",
    "results": "result",
    "chart": "result",
    "plot": "result",
    "graph": "result",
    "concept": "concept",
    "illustration": "concept",
    "schematic": "concept",
}


def normalize_figure_type(name: str) -> str:
    """Map free-form figure type names to the canonical set.

    Raises :class:`ValueError` for genuinely unknown types so the CLI
    surfaces an actionable error rather than dispatching an empty
    prompt to the LLM.
    """
    if not name:
        raise ValueError("figure type is required")
    lowered = name.strip().lower()
    if lowered in FIGURE_TYPES:
        return lowered
    if lowered in _FIGURE_TYPE_ALIASES:
        return _FIGURE_TYPE_ALIASES[lowered]
    raise ValueError(
        f"Unknown figure type: {name!r}. Valid: {', '.join(FIGURE_TYPES)}."
    )


# Per-type variant tables. Each entry is (label, directive) and the
# directive is dropped verbatim into the user prompt so it must be
# meaningful to the LLM, not just to us. Round-robin by index when
# ``n`` exceeds the table length so callers never see an IndexError.
_ARCHITECTURE_VARIANTS: tuple[tuple[str, str], ...] = (
    (
        "layered horizontal",
        "Use a layered horizontal layout: blocks flow left-to-right, "
        "with each layer stacked vertically. Good for encoder-style "
        "or pipeline architectures.",
    ),
    (
        "hub-and-spoke",
        "Use a hub-and-spoke layout: a central core node with "
        "satellite modules connected by arrows. Good for systems "
        "where one component coordinates multiple subsystems.",
    ),
    (
        "encoder-decoder vertical",
        "Use a vertical encoder-decoder split: input flows down "
        "through the encoder on the left half, then up through the "
        "decoder on the right half. Good for sequence-to-sequence "
        "or U-Net-style architectures.",
    ),
)

_RESULT_VARIANTS: tuple[tuple[str, str], ...] = (
    (
        "grouped bar",
        "Render the data as a grouped bar chart with error bars and "
        "a legend in the upper right. Save to 'results_bar.png'.",
    ),
    (
        "line with shaded variance",
        "Render the data as an overlaid line chart with shaded "
        "variance bands (mean ± std). Save to 'results_line.png'.",
    ),
    (
        "paired boxplot",
        "Render the data as side-by-side boxplots (one box per "
        "method, paired by condition). Save to 'results_box.png'.",
    ),
)

_CONCEPT_VARIANTS: tuple[tuple[str, str], ...] = (
    (
        "flat schematic for DALL·E 3",
        "Tune for DALL·E 3: natural-language declarative sentences. "
        "Emphasise clean schematic style, white background, "
        "vector-art aesthetic, left-to-right reading order.",
    ),
    (
        "keyword-heavy Midjourney v6",
        "Tune for Midjourney v6: comma-separated style keywords, "
        "end with `--ar 16:9 --style raw --no text`. Emphasise "
        "academic illustration, minimal palette, isometric or "
        "top-down perspective.",
    ),
    (
        "SD weighted line-art",
        "Tune for Stable Diffusion (SDXL or similar): keyword-heavy, "
        "use weighted emphasis like `(clean line art:1.3)`, "
        "`(white background:1.2)`. Populate negative_prompt with "
        "`photorealistic, cluttered, text, watermark, low quality`.",
    ),
)


_VARIANTS_BY_TYPE: dict[str, tuple[tuple[str, str], ...]] = {
    "architecture": _ARCHITECTURE_VARIANTS,
    "result": _RESULT_VARIANTS,
    "concept": _CONCEPT_VARIANTS,
}

# Output code-language hint that the renderer uses when syntax-
# highlighting the figure. Concept drafts are plain text prompts so
# the "language" is intentionally generic.
_CODE_LANGUAGE_BY_TYPE: dict[str, str] = {
    "architecture": "tikz",
    "result": "python",
    "concept": "text",
}


@dataclass(slots=True)
class FigureDraft:
    """One Illustrator output. Stable shape across all 3 figure types."""

    figure_type: str  # "architecture" | "result" | "concept"
    version: str  # "A" | "B" | "C" | ...
    style_label: str
    code: str  # TikZ source, Python source, or text-to-image prompt
    code_language: str  # "tikz" | "python" | "text"
    notes: str
    suggested_use: str
    target_model: str = ""  # concept-only: "dalle3" | "midjourney" | "sd"
    negative_prompt: str = ""  # concept-only
    extras: dict[str, Any] = field(default_factory=dict)


class Illustrator(BaseAgent):
    """LLM-driven figure code generator."""

    role = "illustrator"

    def __init__(
        self,
        llm: Any,
        *,
        figure_type: str = "architecture",
        language: str = "en",
    ) -> None:
        # We carry ``figure_type`` so :attr:`prompt_name` resolves to
        # the matching YAML; the public ``generate()`` API can still
        # override per-call (it constructs a fresh agent each time
        # the user changes type, since the system prompt is loaded
        # at __init__).
        self._figure_type = normalize_figure_type(figure_type)
        super().__init__(llm, language=language)

    @property
    def prompt_name(self) -> str:
        return f"illustrator_{self._figure_type}"

    @property
    def figure_type(self) -> str:
        return self._figure_type

    def run(self, context: dict[str, Any]) -> AgentResponse:
        """Adapter for the orchestrator. Returns the first draft."""
        description = str(context.get("description", "")).strip()
        if not description:
            raise ValueError("context['description'] is required for the Illustrator")
        drafts = self.generate(
            description,
            data=str(context.get("data", "")),
            n=int(context.get("n", 1)),
            parallel=bool(context.get("parallel", True)),
        )
        primary = drafts[0]
        return AgentResponse(
            role="illustrator",
            content=primary.code,
            metadata={"drafts": drafts},
        )

    def generate(
        self,
        description: str,
        *,
        data: str = "",
        n: int = 2,
        parallel: bool = True,
    ) -> list[FigureDraft]:
        """Produce ``n`` figure drafts.

        ``description`` is the user's free-form ask (e.g. "three-
        layer transformer encoder with feed-forward bypass"). For
        ``result`` figures ``data`` carries the quantitative payload
        (e.g. "accuracy: 85% (ours) vs 80% (baseline)"). Each variant
        cycles through the type-specific directive table so a
        ``--versions 3`` call returns three structurally different
        drafts in parallel.
        """
        if n <= 0:
            return []
        variants_table = _VARIANTS_BY_TYPE[self._figure_type]
        variants = [variants_table[i % len(variants_table)] for i in range(n)]
        versions = [chr(ord("A") + i) for i in range(n)]
        prompts = [
            _build_user_prompt(
                figure_type=self._figure_type,
                description=description,
                data=data,
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

        drafts: list[FigureDraft] = []
        for version, (label, _directive), raw in zip(
            versions, variants, raw_outputs, strict=True
        ):
            parsed = _parse_figure(raw, figure_type=self._figure_type)
            drafts.append(
                FigureDraft(
                    figure_type=self._figure_type,
                    version=version,
                    style_label=parsed.get("style_label") or label,
                    code=parsed.get("code", ""),
                    code_language=_CODE_LANGUAGE_BY_TYPE[self._figure_type],
                    notes=parsed.get("notes", ""),
                    suggested_use=parsed.get("suggested_use", ""),
                    target_model=parsed.get("target_model", ""),
                    negative_prompt=parsed.get("negative_prompt", ""),
                )
            )
        return drafts


def _build_user_prompt(
    *,
    figure_type: str,
    description: str,
    data: str,
    variant_label: str,
    variant_directive: str,
) -> str:
    """Assemble the per-call user prompt.

    The system prompt (YAML) carries the hard rules and output
    schema; this user prompt just supplies the request payload plus
    the variant directive so the LLM knows which slice to lean into.
    """
    lines = [
        f"Figure type: {figure_type}",
        f"User description: {description.strip() or '(none — be generic)'}",
    ]
    if data.strip():
        lines.append(f"Quantitative data / payload: {data.strip()}")
    lines.append(f"Variant directive ({variant_label}): {variant_directive}")
    return "\n".join(lines)


# Required JSON fields for each figure type — used to backfill empty
# strings when the LLM forgets a field. Code is always required.
_REQUIRED_FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "architecture": ("code", "style_label", "notes", "suggested_use"),
    "result": ("code", "style_label", "notes", "suggested_use"),
    "concept": (
        "code",
        "style_label",
        "notes",
        "suggested_use",
        "target_model",
        "negative_prompt",
    ),
}


def _parse_figure(raw: str, *, figure_type: str) -> dict[str, str]:
    """Pull the figure fields out of one Illustrator LLM reply.

    Tolerates: markdown fences, extra prose around the JSON, missing
    optional fields. When the LLM forgets to emit JSON entirely we
    treat the whole reply as the ``code`` payload — better to return
    something rough than to crash.
    """
    try:
        payload = parse_model(raw, FigurePayload)
    except ValueError:
        return {"code": raw.strip(), "style_label": "", "notes": "", "suggested_use": ""}

    fields = _REQUIRED_FIELDS_BY_TYPE.get(figure_type, ("code",))
    out: dict[str, str] = {key: getattr(payload, key, "") for key in fields}
    if not out.get("code"):
        out["code"] = raw.strip()
    return out
