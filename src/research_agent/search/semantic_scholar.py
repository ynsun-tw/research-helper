"""Semantic Scholar Graph API client (no API key needed for basic use).

Two roles:

1. **Search fallback** when arXiv's Atom endpoint 429s our IP. Returns
   :class:`ArxivSearchHit` instances so the rest of the pipeline
   (/search rendering, persistence, Searcher scoring, /read) keeps
   working unchanged.

2. **Citation graph navigation** (M3 T3.1.2.2): forward references via
   :py:meth:`SemanticScholarSearcher.get_citations` and backward
   references via :py:meth:`SemanticScholarSearcher.get_references`.
   Both also return :class:`ArxivSearchHit` so the same render / queue
   plumbing applies.

Hits without an arXiv mapping are filtered out - downstream ``/read``
only knows how to load arXiv ids today.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from research_agent.search.arxiv_search import (
    DEFAULT_MIN_REQUEST_INTERVAL,
    DEFAULT_USER_AGENT,
    ArxivSearchError,
    ArxivSearchHit,
    _retry_after_seconds,
)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_URL = f"{S2_BASE_URL}/paper/search"
S2_PAPER_URL = f"{S2_BASE_URL}/paper"
S2_FIELDS = "title,abstract,year,externalIds,publicationDate"


class SemanticScholarSearcher:
    """Query Semantic Scholar Graph API for papers + their citation graph.

    Mirrors :class:`ArxivSearcher`'s retry / throttle / error semantics
    so callers can swap searchers freely. The 'free' tier limits
    unauthenticated traffic to ~100 requests / 5 min per IP; the same
    3s gap that satisfies arXiv keeps us well under the S2 ceiling too.
    """

    # Process-wide last-request timestamp (monotonic seconds).
    _last_request_at: float = 0.0

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 2,
        retry_backoff: float = 1.5,
        rate_limit_backoff: float = 3.0,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        user_agent: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.rate_limit_backoff = max(0.0, rate_limit_backoff)
        self.min_request_interval = max(0.0, min_request_interval)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.api_key = api_key

    # ------------------------------------------------------------- throttle

    def _await_throttle(self) -> None:
        if self.min_request_interval <= 0:
            return
        last = SemanticScholarSearcher._last_request_at
        if last == 0.0:
            return
        wait = self.min_request_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)

    def _mark_request_done(self) -> None:
        SemanticScholarSearcher._last_request_at = time.monotonic()

    # ----------------------------------------------------------------- HTTP

    def _request_bytes(self, url: str, *, op: str) -> bytes:
        """Fetch ``url`` honouring throttle / retries / 429 backoff.

        ``op`` is a short label ("search", "citations", "references") used
        in error messages so failures stay diagnostic across endpoints.
        """
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        req = urllib.request.Request(url, headers=headers)

        attempts = self.retries + 1
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self._await_throttle()
            self._mark_request_done()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw: bytes = resp.read()
                    return raw
            except TimeoutError as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(self.retry_backoff)
                    continue
                raise ArxivSearchError(
                    f"Semantic Scholar {op} timed out after {self.timeout:.0f}s "
                    f"(retried {self.retries}x)."
                ) from exc
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429 and attempt < attempts:
                    delay = _retry_after_seconds(exc.headers.get("Retry-After"))
                    if delay is None:
                        delay = self.rate_limit_backoff * (2 ** (attempt - 1))
                    time.sleep(delay)
                    continue
                if exc.code == 429:
                    raise ArxivSearchError(
                        f"Semantic Scholar is rate-limiting this client "
                        f"on /{op} (HTTP 429). Retried {self.retries}x; "
                        "consider obtaining a free API key from "
                        "https://www.semanticscholar.org/product/api"
                    ) from exc
                if exc.code == 404:
                    raise ArxivSearchError(
                        f"Semantic Scholar has no record for this paper "
                        f"({op}, url={url})"
                    ) from exc
                raise ArxivSearchError(
                    f"Semantic Scholar {op} failed: HTTP {exc.code} "
                    f"{exc.reason} (url={url})"
                ) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError) and attempt < attempts:
                    last_exc = exc
                    time.sleep(self.retry_backoff)
                    continue
                raise ArxivSearchError(
                    f"Semantic Scholar {op} failed: {exc}"
                ) from exc

        raise ArxivSearchError(
            f"Semantic Scholar {op} failed after {attempts} attempt(s): "
            f"{last_exc}"
        )

    # --------------------------------------------------------------- search

    def search(self, query: str, *, max_results: int = 5) -> list[ArxivSearchHit]:
        params = {
            "query": query.strip(),
            "limit": str(_clamp_limit(max_results)),
            "fields": S2_FIELDS,
        }
        url = f"{S2_API_URL}?{urllib.parse.urlencode(params)}"
        try:
            return _parse_s2_response(self._request_bytes(url, op="search"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArxivSearchError(
                f"Semantic Scholar returned an unparseable response: {exc}"
            ) from exc

    # ----------------------------------------------------- citation graph

    def get_citations(
        self, arxiv_id: str, *, max_results: int = 10
    ) -> list[ArxivSearchHit]:
        """Papers that cite ``arxiv_id`` (forward references).

        Useful for "who built on this?" follow-up reading.
        """
        return self._citation_relation(
            arxiv_id, endpoint="citations", key="citingPaper", max_results=max_results
        )

    def get_references(
        self, arxiv_id: str, *, max_results: int = 10
    ) -> list[ArxivSearchHit]:
        """Papers cited by ``arxiv_id`` (backward references).

        Useful for understanding what a paper builds on.
        """
        return self._citation_relation(
            arxiv_id, endpoint="references", key="citedPaper", max_results=max_results
        )

    def _citation_relation(
        self,
        arxiv_id: str,
        *,
        endpoint: str,
        key: str,
        max_results: int,
    ) -> list[ArxivSearchHit]:
        paper_id = _format_paper_id(arxiv_id)
        params = {"limit": str(_clamp_limit(max_results)), "fields": S2_FIELDS}
        url = f"{S2_PAPER_URL}/{paper_id}/{endpoint}?{urllib.parse.urlencode(params)}"
        try:
            return _parse_s2_relations(
                self._request_bytes(url, op=endpoint), key=key
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArxivSearchError(
                f"Semantic Scholar returned an unparseable response: {exc}"
            ) from exc


# --------------------------------------------------------------- helpers


def _clamp_limit(n: int) -> int:
    """S2 caps /paper/search at 100 and /paper/.../citations at 1000.
    1-100 is the safe shared range we use for both."""
    return max(1, min(int(n), 100))


def _format_paper_id(arxiv_id: str) -> str:
    """S2 accepts ``ARXIV:<id>`` for arXiv-keyed lookups.

    URL-encode the whole token so old-format ids like ``cs.CL/0001001``
    (which contain a slash) survive being placed in a URL path.
    """
    stripped = arxiv_id.strip()
    if not stripped:
        raise ArxivSearchError("arxiv_id is required")
    return urllib.parse.quote(f"ARXIV:{stripped}", safe="")


def _paper_dict_to_hit(entry: object) -> ArxivSearchHit | None:
    """Convert an S2 paper dict into a hit; return None if no arXiv mapping."""
    if not isinstance(entry, dict):
        return None
    external = entry.get("externalIds") or {}
    if not isinstance(external, dict):
        external = {}
    arxiv_id = (
        external.get("ArXiv") or external.get("arXiv") or external.get("arxiv")
    )
    if not arxiv_id:
        return None
    title = (entry.get("title") or "").strip()
    abstract = (entry.get("abstract") or "").strip()
    published = entry.get("publicationDate") or ""
    if not published:
        year = entry.get("year")
        if isinstance(year, int):
            published = f"{year}-01-01"
    return ArxivSearchHit(
        arxiv_id=str(arxiv_id).strip(),
        title=title or "(untitled)",
        abstract=abstract,
        published=str(published),
        source="semantic_scholar",
    )


def _parse_s2_response(data: bytes) -> list[ArxivSearchHit]:
    """Parse ``/paper/search`` payload into hits, dropping non-arXiv entries."""
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ArxivSearchError("Semantic Scholar response was not a JSON object")
    entries = payload.get("data") or []
    if not isinstance(entries, list):
        raise ArxivSearchError("Semantic Scholar response missing 'data' list")
    return [h for h in (_paper_dict_to_hit(e) for e in entries) if h is not None]


def _parse_s2_relations(data: bytes, *, key: str) -> list[ArxivSearchHit]:
    """Parse ``/paper/{id}/citations|references`` payload.

    Each entry in the ``data`` list wraps the related paper under
    ``key`` (``citingPaper`` for ``/citations``, ``citedPaper`` for
    ``/references``).
    """
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ArxivSearchError("Semantic Scholar response was not a JSON object")
    entries = payload.get("data") or []
    if not isinstance(entries, list):
        raise ArxivSearchError("Semantic Scholar response missing 'data' list")
    hits: list[ArxivSearchHit] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        hit = _paper_dict_to_hit(entry.get(key))
        if hit is not None:
            hits.append(hit)
    return hits


__all__ = [
    "S2_API_URL",
    "S2_BASE_URL",
    "S2_PAPER_URL",
    "SemanticScholarSearcher",
    "_format_paper_id",
    "_paper_dict_to_hit",
    "_parse_s2_relations",
    "_parse_s2_response",
]
