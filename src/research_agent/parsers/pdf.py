"""PyMuPDF-based PDF parser producing a :class:`Paper`.

The heuristics here are intentionally simple — high-quality structured
extraction is out of scope for the MVP. We target reasonable defaults that
work for typical arXiv-style two-column or single-column PDFs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz

from research_agent.core.paper import Paper, Section


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed."""


_SECTION_NAMES = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "appendix",
)

_NUMBERED_HEAD = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s+([A-Z][A-Za-z0-9 \-:,&/]{2,80})\s*$")
_NAMED_HEAD = re.compile(
    r"^\s*(" + "|".join(re.escape(n) for n in _SECTION_NAMES) + r")\s*$",
    re.IGNORECASE,
)


def parse_pdf(path: Path, *, paper_id: str | None = None) -> Paper:
    """Parse a local PDF into a :class:`Paper`."""
    if not path.exists():
        raise PDFParseError(f"PDF not found: {path}")
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PDFParseError(f"Failed to open PDF: {path}") from exc

    try:
        title = _extract_title(doc)
        full_text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()

    sections = _split_sections(full_text)
    abstract = _extract_abstract(sections, full_text)
    authors = _extract_authors(full_text, title)
    pid = paper_id or f"local:{_file_fingerprint(path)}"

    return Paper(
        id=pid,
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
        full_text=full_text,
        pdf_path=path,
    )


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _extract_title(doc: fitz.Document) -> str:
    """Pick the span with the largest font size on the first page."""
    if doc.page_count == 0:
        return ""
    page = doc[0]
    data = page.get_text("dict")
    best_size = 0.0
    best_text = ""
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = float(span.get("size", 0))
                text = span.get("text", "").strip()
                if not text or len(text) < 4:
                    continue
                if size > best_size:
                    best_size = size
                    best_text = text
    return best_text or _first_nonempty_line(page.get_text("text"))


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _split_sections(text: str) -> list[Section]:
    """Split full text into sections by heading heuristics."""
    lines = text.splitlines()
    boundaries: list[tuple[int, str]] = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or len(line) > 80:
            continue
        named = _NAMED_HEAD.match(line)
        numbered = _NUMBERED_HEAD.match(line)
        if named:
            boundaries.append((i, named.group(1).strip().title()))
        elif numbered:
            boundaries.append((i, f"{numbered.group(1)} {numbered.group(2)}".strip()))

    if not boundaries:
        return []

    sections: list[Section] = []
    for idx, (start, title) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        sections.append(Section(title=title, content=body))
    return sections


def _extract_abstract(sections: list[Section], full_text: str) -> str:
    for s in sections:
        if s.title.lower().startswith("abstract"):
            return s.content
    lower = full_text.lower()
    start = lower.find("abstract")
    if start == -1:
        return ""
    snippet = full_text[start + len("abstract") :].lstrip(" :\n\t")
    end_markers = ("\n1 ", "\n1. ", "\nIntroduction", "\nINTRODUCTION", "\nKeywords")
    cut = len(snippet)
    for marker in end_markers:
        m = snippet.find(marker)
        if m != -1:
            cut = min(cut, m)
    return snippet[:cut].strip()


def _extract_authors(full_text: str, title: str) -> list[str]:
    """Best-effort: take the line(s) immediately after the title, before Abstract."""
    if not title:
        return []
    idx = full_text.find(title)
    if idx == -1:
        return []
    tail = full_text[idx + len(title) :]
    abstract_idx = tail.lower().find("abstract")
    candidate = tail[:abstract_idx] if abstract_idx != -1 else tail[:400]
    names: list[str] = []
    for raw in candidate.splitlines():
        line = raw.strip()
        if not line or len(line) > 200:
            continue
        if any(ch.isdigit() for ch in line) and "@" not in line:
            continue
        parts = [p.strip() for p in re.split(r",| and ", line) if p.strip()]
        for p in parts:
            if 2 <= len(p) <= 60 and " " in p and p[0].isupper():
                names.append(p)
        if names:
            break
    return names
