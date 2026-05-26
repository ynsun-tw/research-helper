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

# arXiv asks polite clients to identify themselves with a contact URL and
# limit requests to one per ~3 seconds. We honour both by default.
DEFAULT_USER_AGENT = (
    "research-agent/0.1 (+https://github.com/research-agent; "
    "polite-cli-multi-agent-research-assistant)"
)
DEFAULT_MIN_REQUEST_INTERVAL = 3.0


class ArxivSearchError(Exception):
    """Raised when arXiv search fails."""


@dataclass(slots=True)
class ArxivSearchHit:
    """A single paper hit. ``source`` records which engine produced it
    (``arxiv`` or ``semantic_scholar``) so the UI can flag fallbacks.

    The historical name ``ArxivSearchHit`` is kept for backwards
    compatibility; non-arXiv sources still emit instances of this class
    but only ever populate ``arxiv_id`` for papers that have an arXiv
    mapping (so downstream ``/read`` still works).
    """

    arxiv_id: str
    title: str
    abstract: str
    published: str = ""
    relevance_score: float | None = None
    relevance_reason: str = ""
    source: str = "arxiv"


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

    Politeness: a class-level monotonic clock ensures every actual HTTP
    request is spaced by at least ``min_request_interval`` seconds (default
    3.0, per arXiv TOS). The state is class-level so the typical
    ``ArxivSearcher().search(...)`` usage in :mod:`paper_resolver` -
    which builds a fresh instance per call - still honours the gap.
    Tests pass ``min_request_interval=0`` to opt out.
    """

    # Process-wide last-request timestamp (monotonic seconds).
    _last_request_at: float = 0.0

    def __init__(
        self,
        *,
        timeout: float = 45.0,
        retries: int = 2,
        retry_backoff: float = 1.5,
        rate_limit_backoff: float = 3.0,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        user_agent: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.rate_limit_backoff = max(0.0, rate_limit_backoff)
        self.min_request_interval = max(0.0, min_request_interval)
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    def _await_throttle(self) -> None:
        """Sleep just long enough that the next request respects the TOS gap.

        The very first request in a process makes no wait - the gap only
        applies *between* requests. ``_last_request_at == 0.0`` is the
        "never called" sentinel.
        """
        if self.min_request_interval <= 0:
            return
        last = ArxivSearcher._last_request_at
        if last == 0.0:
            return
        wait = self.min_request_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)

    def _mark_request_done(self) -> None:
        ArxivSearcher._last_request_at = time.monotonic()

    def search(self, query: str, *, max_results: int = 5) -> list[ArxivSearchHit]:
        q = urllib.parse.quote(f"all:{query.strip()}")
        url = f"{ARXIV_API_URL}?search_query={q}&start=0&max_results={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

        attempts = self.retries + 1
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self._await_throttle()
            # Record intent-to-call *before* urlopen so timeouts / 429s
            # still count toward the next request's gap.
            self._mark_request_done()
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
