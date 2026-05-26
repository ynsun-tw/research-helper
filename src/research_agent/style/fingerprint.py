"""Dataclasses + JSON (de)serialization for the style fingerprint.

Three layers, mirroring milestone S4.1.2:

- :class:`MacroFingerprint`: structural signals visible at the
  section level (abstract opener, intro arc, related-work
  organization). Heuristic today; an LLM-augmented summary slot
  (``notes``) is reserved for S4.1.3.
- :class:`MicroFingerprint`: pure statistical signals - sentence
  length distribution, transition / hedging / confidence word rates,
  passive-voice rate, type-token ratio.
- :class:`PersonalMarkers`: surface markers - section naming, citation
  style, figure/table references, em-dash usage.

All three are wrapped in :class:`Fingerprint` and persisted as JSON
at ``~/.research-agent/style/fingerprint.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MacroFingerprint:
    """Section-level structural signals."""

    abstract_opener: str = ""
    abstract_avg_sentences: float = 0.0
    intro_opener: str = ""
    intro_avg_paragraphs: float = 0.0
    related_work_strategy: str = ""
    section_count_avg: float = 0.0
    notes: str = ""


@dataclass(slots=True)
class MicroFingerprint:
    """Statistical signals computed from raw paragraphs."""

    avg_sentence_length: float = 0.0
    median_sentence_length: float = 0.0
    p10_sentence_length: float = 0.0
    p90_sentence_length: float = 0.0
    avg_paragraph_length: float = 0.0  # sentences per paragraph
    sentence_count: int = 0
    transition_freq: dict[str, float] = field(default_factory=dict)
    hedging_per_100: float = 0.0
    confidence_per_100: float = 0.0
    passive_per_100: float = 0.0
    type_token_ratio: float = 0.0


@dataclass(slots=True)
class PersonalMarkers:
    """Surface conventions: how the author labels their figures and cites."""

    top_section_titles: list[str] = field(default_factory=list)
    citation_format: str = ""  # "latex_cite" | "bracket_num" | "author_year" | "mixed" | ""
    figure_ref_format: str = ""  # "Figure" | "Fig." | ""
    table_ref_format: str = ""  # "Table" | "Tab." | ""
    em_dash_per_100: float = 0.0


@dataclass(slots=True)
class Fingerprint:
    """The composite style fingerprint persisted to disk."""

    macro: MacroFingerprint = field(default_factory=MacroFingerprint)
    micro: MicroFingerprint = field(default_factory=MicroFingerprint)
    markers: PersonalMarkers = field(default_factory=PersonalMarkers)
    sample_count: int = 0
    paper_count: int = 0
    version: int = 1
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fingerprint:
        macro_data = data.get("macro", {}) or {}
        micro_data = data.get("micro", {}) or {}
        markers_data = data.get("markers", {}) or {}
        return cls(
            macro=MacroFingerprint(**_filter_known_keys(macro_data, MacroFingerprint)),
            micro=MicroFingerprint(**_filter_known_keys(micro_data, MicroFingerprint)),
            markers=PersonalMarkers(**_filter_known_keys(markers_data, PersonalMarkers)),
            sample_count=int(data.get("sample_count", 0) or 0),
            paper_count=int(data.get("paper_count", 0) or 0),
            version=int(data.get("version", 1) or 1),
            created_at=str(data.get("created_at", "") or ""),
        )

    def save_to(self, path: Path, *, preserve_history: bool = False) -> None:
        """Persist the fingerprint JSON.

        When ``preserve_history`` is true and ``path`` already exists,
        archive the existing file to ``<stem>_v<old_version>.json``
        beside it and bump ``self.version`` to ``old_version + 1``.
        This keeps a linear history (``fingerprint_v1.json``,
        ``fingerprint_v2.json``, …) next to the always-current
        ``fingerprint.json``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if preserve_history and path.exists():
            try:
                old = Fingerprint.load_from(path)
                archived = path.with_name(f"{path.stem}_v{old.version}{path.suffix}")
                # If the archive slot is already taken (re-run with same
                # version), just append an incrementing suffix to avoid
                # silently clobbering it.
                idx = 0
                while archived.exists():
                    idx += 1
                    archived = path.with_name(
                        f"{path.stem}_v{old.version}-{idx}{path.suffix}"
                    )
                path.rename(archived)
                self.version = old.version + 1
            except Exception:
                # If the existing file is unreadable, we'd rather
                # overwrite it than crash the update.
                pass
        if not self.created_at:
            self.created_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load_from(cls, path: Path) -> Fingerprint:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid fingerprint payload in {path}")
        return cls.from_dict(data)


def _filter_known_keys(data: dict[str, Any], cls: type) -> dict[str, Any]:
    """Drop unknown keys so future JSON files don't break older code, and
    vice versa - old JSON files can still hydrate newer dataclasses."""
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in known}
