"""arXiv Atom API search (no API key)."""

from __future__ import annotations

import re
import time
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
    """Query export.arxiv.org for papers by title/keyword.

    Wraps socket / URL errors as :class:`ArxivSearchError` so callers can
    catch a single exception type. Retries are intentionally narrow:

    - read timeout (``TimeoutError`` or ``URLError(reason=TimeoutError)``)
      -> retry with ``retry_backoff`` linear delay
    - HTTP 429 Too Many Requests -> retry honouring ``Retry-After`` when
      present, otherwise exponential backoff seeded from
      ``rate_limit_backoff``
    - everything else (DNS, refused connection, 4xx other than 429, 5xx
      that doesn't recur) fails fast - retries don't fix those.
    """

    def __init__(
        self,
        *,
        timeout: float = 45.0,
        retries: int = 2,
        retry_backoff: float = 1.5,
        rate_limit_backoff: float = 3.0,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.rate_limit_backoff = max(0.0, rate_limit_backoff)

    def search(self, query: str, *, max_results: int = 5) -> list[ArxivSearchHit]:
        q = urllib.parse.quote(f"all:{query.strip()}")
        url = f"{ARXIV_API_URL}?search_query={q}&start=0&max_results={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "research-agent/0.1"})

        attempts = self.retries + 1
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                return _parse_atom_feed(data)
            except TimeoutError as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(self.retry_backoff)
                    continue
                raise ArxivSearchError(
                    f"arXiv search timed out after {self.timeout:.0f}s "
                    f"(retried {self.retries}x). The API may be slow; "
                    "try again in a moment or tighten your query."
                ) from exc
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429 and attempt < attempts:
                    delay = _retry_after_seconds(exc.headers.get("Retry-After"))
                    if delay is None:
                        # arXiv asks for ~3s between requests; back off
                        # exponentially across retries.
                        delay = self.rate_limit_backoff * (2 ** (attempt - 1))
                    time.sleep(delay)
                    continue
                if exc.code == 429:
                    raise ArxivSearchError(
                        "arXiv is rate-limiting this client (HTTP 429). "
                        f"Retried {self.retries}x; wait ~30s before searching "
                        "again, or reduce how often you call /search."
                    ) from exc
                raise ArxivSearchError(
                    f"arXiv search failed: HTTP {exc.code} {exc.reason} "
                    f"(url={url})"
                ) from exc
            except urllib.error.URLError as exc:
                # urllib sometimes wraps socket timeouts here.
                if isinstance(exc.reason, TimeoutError) and attempt < attempts:
                    last_exc = exc
                    time.sleep(self.retry_backoff)
                    continue
                raise ArxivSearchError(f"arXiv search failed: {exc}") from exc
            except (ET.ParseError, ValueError) as exc:
                raise ArxivSearchError(
                    f"arXiv returned an unparseable response: {exc}"
                ) from exc

        # Defensive: loop above always returns or raises; keep mypy happy.
        raise ArxivSearchError(
            f"arXiv search failed after {attempts} attempt(s): {last_exc}"
        )


def _retry_after_seconds(header: str | None) -> float | None:
    """Parse a Retry-After header value into seconds, or None if absent/bad.

    Supports the plain-seconds form (e.g. "30"). The HTTP-date form is
    ignored - arXiv only emits the seconds form in practice and falling
    back to the configured exponential backoff is safer than parsing
    RFC dates here.
    """
    if not header:
        return None
    try:
        seconds = float(header.strip())
    except (TypeError, ValueError):
        return None
    # Clamp to a sane range so a buggy server can't stall us for hours.
    if seconds < 0:
        return None
    return min(seconds, 60.0)


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
