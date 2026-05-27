"""Paper entity used across parsers, storage, and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    """A logical section of a paper (e.g. Introduction, Methods)."""

    title: str
    content: str


@dataclass
class Paper:
    """Structured representation of a research paper.

    `id` conventions:
      - `arxiv:<id>` for papers downloaded from arXiv
      - `local:<sha1-prefix>` for locally provided PDFs
    """

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    full_text: str = ""
    year: int | None = None
    venue: str | None = None
    pdf_path: Path | None = None
    tags: list[str] = field(default_factory=list)

    def section_titles(self) -> list[str]:
        return [s.title for s in self.sections]
