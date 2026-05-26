"""Semantic Scholar Graph API search (no API key needed for basic use).

Used as a fallback when arXiv's Atom endpoint 429s our IP. Returns
:class:`ArxivSearchHit` instances so the rest of the pipeline
(/search rendering, persistence, Searcher scoring, /read) keeps working
unchanged. Hits without an arXiv mapping are filtered out - downstream
``/read`` only knows how to load arXiv ids today.
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

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,externalIds,publicationDate"


class SemanticScholarSearcher:
    """Query Semantic Scholar Graph API for papers by title/keyword.

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

    # --------------------------------------------------------------- search

    def search(self, query: str, *, max_results: int = 5) -> list[ArxivSearchHit]:
        params = {
            "query": query.strip(),
            "limit": str(max(1, min(max_results, 100))),
            "fields": S2_FIELDS,
        }
        url = f"{S2_API_URL}?{urllib.parse.urlencode(params)}"
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
                    data = resp.read()
                return _parse_s2_response(data)
            except TimeoutError as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(self.retry_backoff)
                    continue
                raise ArxivSearchError(
                    f"Semantic Scholar timed out after {self.timeout:.0f}s "
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
                        "Semantic Scholar is rate-limiting this client "
                        f"(HTTP 429). Retried {self.retries}x; consider "
                        "obtaining a free API key from "
                        "https://www.semanticscholar.org/product/api"
                    ) from exc
                raise ArxivSearchError(
                    f"Semantic Scholar search failed: HTTP {exc.code} "
                    f"{exc.reason} (url={url})"
                ) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError) and attempt < attempts:
                    last_exc = exc
                    time.sleep(self.retry_backoff)
                    continue
                raise ArxivSearchError(
                    f"Semantic Scholar search failed: {exc}"
                ) from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise ArxivSearchError(
                    f"Semantic Scholar returned an unparseable response: {exc}"
                ) from exc

        raise ArxivSearchError(
            f"Semantic Scholar search failed after {attempts} attempt(s): "
            f"{last_exc}"
        )


def _parse_s2_response(data: bytes) -> list[ArxivSearchHit]:
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
        external = entry.get("externalIds") or {}
        if not isinstance(external, dict):
            external = {}
        arxiv_id = external.get("ArXiv") or external.get("arXiv") or external.get("arxiv")
        if not arxiv_id:
            # Skip papers without an arXiv mapping - /read can't load them.
            continue
        title = (entry.get("title") or "").strip()
        abstract = (entry.get("abstract") or "").strip()
        published = entry.get("publicationDate") or ""
        if not published:
            year = entry.get("year")
            if isinstance(year, int):
                published = f"{year}-01-01"
        hits.append(
            ArxivSearchHit(
                arxiv_id=str(arxiv_id).strip(),
                title=title or "(untitled)",
                abstract=abstract,
                published=str(published),
                source="semantic_scholar",
            )
        )
    return hits


__all__ = [
    "S2_API_URL",
    "SemanticScholarSearcher",
    "_parse_s2_response",
]
