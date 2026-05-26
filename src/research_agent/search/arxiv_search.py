"""arXiv Atom API search (no API key)."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from research_agent.search.arxiv import normalize_arxiv_id

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([^/\s]+)", re.IGNORECASE)


class ArxivSearchError(Exception):
    """Raised when arXiv search fails."""


@dataclass(slots=True)
class ArxivSearchHit:
    arxiv_id: str
    title: str
    abstract: str
    published: str = ""
    relevance_score: float | None = None
    relevance_reason: str = ""


class ArxivSearcher:
    """Query export.arxiv.org for papers by title/keyword."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> list[ArxivSearchHit]:
        q = urllib.parse.quote(f"all:{query.strip()}")
        url = f"{ARXIV_API_URL}?search_query={q}&start=0&max_results={max_results}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-agent/0.1"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.URLError as exc:
            raise ArxivSearchError(f"arXiv search failed: {exc}") from exc
        return _parse_atom_feed(data)


def _parse_atom_feed(xml_bytes: bytes) -> list[ArxivSearchHit]:
    root = ET.fromstring(xml_bytes)
    hits: list[ArxivSearchHit] = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        raw_id = (entry.findtext(f"{{{ATOM_NS}}}id") or "").strip()
        arxiv_id = _arxiv_id_from_entry_id(raw_id)
        if not arxiv_id:
            continue
        title = _clean_text(entry.findtext(f"{{{ATOM_NS}}}title") or "")
        abstract = _clean_text(entry.findtext(f"{{{ATOM_NS}}}summary") or "")
        published = (entry.findtext(f"{{{ATOM_NS}}}published") or "")[:10]
        hits.append(
            ArxivSearchHit(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                published=published,
            )
        )
    return hits


def _arxiv_id_from_entry_id(entry_id: str) -> str | None:
    m = ARXIV_ABS_RE.search(entry_id)
    if m:
        return normalize_arxiv_id(m.group(1)) or m.group(1)
    return normalize_arxiv_id(entry_id)


def _clean_text(text: str) -> str:
    return " ".join(text.split())
