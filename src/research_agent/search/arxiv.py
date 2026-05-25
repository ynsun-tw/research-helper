"""arXiv PDF fetcher with local caching."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

ARXIV_ID_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(v\d+)?)$", re.IGNORECASE)
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"


class ArxivFetchError(Exception):
    """Raised when fetching from arXiv fails."""


def normalize_arxiv_id(source: str) -> str | None:
    """Return the canonical arXiv id (e.g. '2301.12345') or None if not arXiv."""
    m = ARXIV_ID_RE.match(source.strip())
    return m.group(1) if m else None


class ArxivFetcher:
    """Download arXiv PDFs into a local cache directory."""

    def __init__(self, cache_dir: Path, *, timeout: float = 30.0) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout

    def cache_path(self, arxiv_id: str) -> Path:
        return self.cache_dir / f"{arxiv_id}.pdf"

    def fetch(self, source: str, *, force: bool = False) -> Path:
        """Download `arxiv:<id>` (or bare id) into the cache, return the file path."""
        arxiv_id = normalize_arxiv_id(source)
        if arxiv_id is None:
            raise ArxivFetchError(f"Not a valid arXiv id: {source!r}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(arxiv_id)
        if path.exists() and not force:
            return path

        url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        try:
            self._download(url, path)
        except urllib.error.URLError as exc:
            raise ArxivFetchError(f"Failed to download {url}: {exc}") from exc
        return path

    def _download(self, url: str, dest: Path) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "research-agent/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
