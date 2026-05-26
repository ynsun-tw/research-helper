"""Unit tests for arXiv Atom search parsing."""

from __future__ import annotations

import http.client
import io
import urllib.error

import pytest

from research_agent.search.arxiv_search import (
    ArxivSearcher,
    ArxivSearchError,
    _parse_atom_feed,
    _retry_after_seconds,
)


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = http.client.HTTPMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        url="https://export.arxiv.org/api/query",
        code=code,
        msg="Too Many Requests" if code == 429 else "Error",
        hdrs=headers,
        fp=io.BytesIO(b""),
    )


@pytest.fixture(autouse=True)
def _reset_arxiv_throttle() -> None:
    """Reset the class-level throttle timestamp before every test."""
    ArxivSearcher._last_request_at = 0.0

SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>Sequence models based on complex recurrent networks.</summary>
    <published>2017-06-12T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.12345</id>
    <title>Another Paper</title>
    <summary>Abstract two.</summary>
    <published>2023-01-15T00:00:00Z</published>
  </entry>
</feed>
"""


def test_parse_atom_feed() -> None:
    hits = _parse_atom_feed(SAMPLE_FEED)
    assert len(hits) == 2
    assert hits[0].arxiv_id == "1706.03762v5"
    assert "Attention" in hits[0].title
    assert hits[0].published.startswith("2017")


def test_search_uses_http(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    hits = ArxivSearcher(min_request_interval=0).search(
        "attention transformer", max_results=5
    )
    assert hits[0].arxiv_id.startswith("1706")


def test_search_wraps_socket_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw TimeoutError must become ArxivSearchError, not bubble out."""

    def boom(*_a: object, **_kw: object) -> object:
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", boom
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep", lambda _s: None
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        ArxivSearcher(timeout=1.0, retries=0, min_request_interval=0).search(
            "anything"
        )
    assert "timed out" in str(exc_info.value)


def test_search_retries_once_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient timeout should be retried before giving up."""
    calls: list[int] = []

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(*_a: object, **_kw: object) -> FakeResp:
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("read operation timed out")
        return FakeResp()

    sleeps: list[float] = []
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep", lambda s: sleeps.append(s)
    )
    hits = ArxivSearcher(
        timeout=1.0, retries=1, retry_backoff=0.05, min_request_interval=0
    ).search("x")
    assert len(hits) == 2
    assert len(calls) == 2
    assert sleeps == [0.05]


def test_search_retries_url_error_when_reason_is_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urllib wraps some timeouts inside URLError(reason=TimeoutError(...))."""
    calls: list[int] = []

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(*_a: object, **_kw: object) -> FakeResp:
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError(TimeoutError("timed out"))
        return FakeResp()

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep", lambda _s: None
    )
    hits = ArxivSearcher(
        timeout=1.0, retries=1, retry_backoff=0.0, min_request_interval=0
    ).search("x")
    assert len(hits) == 2
    assert len(calls) == 2


def test_search_non_timeout_url_error_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS / connection-refused style errors should NOT be retried."""
    calls: list[int] = []

    def fake_urlopen(*_a: object, **_kw: object) -> object:
        calls.append(1)
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        ArxivSearcher(timeout=1.0, retries=3, min_request_interval=0).search("x")
    assert len(calls) == 1
    assert "Name or service not known" in str(exc_info.value)


def test_search_retries_on_http_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 429 must trigger retry with backoff, not fail on the first hit."""
    calls: list[int] = []

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(*_a: object, **_kw: object) -> FakeResp:
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429)
        return FakeResp()

    sleeps: list[float] = []
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep", lambda s: sleeps.append(s)
    )
    hits = ArxivSearcher(
        retries=2, rate_limit_backoff=2.0, min_request_interval=0
    ).search("x")
    assert len(hits) == 2
    assert calls == [1, 1]
    # First retry uses base backoff (2 * 2**0 = 2.0).
    assert sleeps == [2.0]


def test_search_429_honours_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(*_a: object, **_kw: object) -> FakeResp:
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429, retry_after="5")
        return FakeResp()

    sleeps: list[float] = []
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep", lambda s: sleeps.append(s)
    )
    ArxivSearcher(
        retries=1, rate_limit_backoff=0.1, min_request_interval=0
    ).search("x")
    assert sleeps == [5.0]


def test_search_429_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(429)),
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep", lambda _s: None
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        ArxivSearcher(
            retries=2, rate_limit_backoff=0.0, min_request_interval=0
        ).search("x")
    msg = str(exc_info.value)
    assert "rate-limit" in msg
    assert "HTTP 429" in msg or "429" in msg


def test_search_non_429_http_error_includes_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500 should fail fast with a debuggable message including the URL."""
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(500)),
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        ArxivSearcher(retries=2, min_request_interval=0).search("transformer")
    msg = str(exc_info.value)
    assert "HTTP 500" in msg
    assert "export.arxiv.org" in msg


def test_throttle_blocks_second_request_within_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two back-to-back searches must sleep to respect arXiv's 3s gap."""

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    # Fake clock anchored at 100.0 so we never collide with the 0.0
    # "never called" sentinel. Calls don't advance time, so the gap
    # between requests is always 0.
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.monotonic", lambda: 100.0
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep",
        lambda s: sleeps.append(s),
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    s = ArxivSearcher(min_request_interval=3.0, retries=0)
    s.search("a")
    s.search("b")
    # First call: sentinel == 0.0 so no wait. Second call: last=100,
    # now=100, elapsed=0 < 3.0 so sleeps the full interval.
    assert sleeps == [3.0]


def test_throttle_disabled_when_interval_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    sleeps: list[float] = []
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.time.sleep",
        lambda s: sleeps.append(s),
    )
    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    s = ArxivSearcher(min_request_interval=0.0, retries=0)
    s.search("a")
    s.search("b")
    assert sleeps == []


def test_default_user_agent_identifies_research_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """arXiv 429s anonymous traffic more aggressively; advertise who we are."""
    captured: dict[str, str] = {}

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: urllib.request.Request, *_a: object, **_kw: object) -> FakeResp:
        captured["ua"] = req.get_header("User-agent")
        return FakeResp()

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    ArxivSearcher(min_request_interval=0).search("x")
    ua = captured["ua"]
    assert "research-agent" in ua
    assert "github.com" in ua  # contact url present


def test_custom_user_agent_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeResp:
        def read(self) -> bytes:
            return SAMPLE_FEED

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(req: urllib.request.Request, *_a: object, **_kw: object) -> FakeResp:
        captured["ua"] = req.get_header("User-agent")
        return FakeResp()

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    ArxivSearcher(min_request_interval=0, user_agent="custom-bot/1.0").search("x")
    assert captured["ua"] == "custom-bot/1.0"


def test_retry_after_parser() -> None:
    assert _retry_after_seconds(None) is None
    assert _retry_after_seconds("") is None
    assert _retry_after_seconds("not a number") is None
    assert _retry_after_seconds("-3") is None
    assert _retry_after_seconds("5") == 5.0
    assert _retry_after_seconds(" 10 ") == 10.0
    # Clamps absurd values.
    assert _retry_after_seconds("3600") == 60.0


def test_search_wraps_xml_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def read(self) -> bytes:
            return b"not xml at all"

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "research_agent.search.arxiv_search.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        ArxivSearcher(min_request_interval=0).search("x")
    assert "unparseable" in str(exc_info.value)
