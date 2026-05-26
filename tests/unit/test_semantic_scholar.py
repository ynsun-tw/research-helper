"""Tests for the Semantic Scholar fallback searcher (M3 task C)."""

from __future__ import annotations

import http.client
import io
import json
import urllib.error

import pytest

from research_agent.search.arxiv_search import ArxivSearchError
from research_agent.search.semantic_scholar import (
    SemanticScholarSearcher,
    _parse_s2_response,
)


def _s2_payload(entries: list[dict[str, object]]) -> bytes:
    return json.dumps(
        {"total": len(entries), "offset": 0, "data": entries}
    ).encode("utf-8")


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = http.client.HTTPMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        url="https://api.semanticscholar.org/graph/v1/paper/search",
        code=code,
        msg="Too Many Requests" if code == 429 else "Error",
        hdrs=headers,
        fp=io.BytesIO(b""),
    )


@pytest.fixture(autouse=True)
def _reset_s2_throttle() -> None:
    SemanticScholarSearcher._last_request_at = 0.0


# ----------------------------- parser ------------------------------


def test_parse_extracts_arxiv_id_from_external_ids() -> None:
    hits = _parse_s2_response(
        _s2_payload(
            [
                {
                    "paperId": "abc",
                    "title": "Attention Is All You Need",
                    "abstract": "Sequence transduction…",
                    "year": 2017,
                    "externalIds": {"ArXiv": "1706.03762", "DOI": "10.x/y"},
                    "publicationDate": "2017-06-12",
                },
            ]
        )
    )
    assert len(hits) == 1
    h = hits[0]
    assert h.arxiv_id == "1706.03762"
    assert h.title == "Attention Is All You Need"
    assert h.published == "2017-06-12"
    assert h.source == "semantic_scholar"


def test_parse_skips_entries_without_arxiv_id() -> None:
    hits = _parse_s2_response(
        _s2_payload(
            [
                {
                    "paperId": "no-arxiv",
                    "title": "Pure conference paper",
                    "externalIds": {"DOI": "10.x/y"},
                    "year": 2020,
                },
                {
                    "paperId": "yes-arxiv",
                    "title": "With arXiv",
                    "externalIds": {"ArXiv": "2005.14165"},
                    "year": 2020,
                },
            ]
        )
    )
    assert [h.arxiv_id for h in hits] == ["2005.14165"]


def test_parse_synthesises_published_from_year() -> None:
    hits = _parse_s2_response(
        _s2_payload(
            [
                {
                    "title": "Foo",
                    "year": 2019,
                    "externalIds": {"ArXiv": "1901.00001"},
                },
            ]
        )
    )
    assert hits[0].published == "2019-01-01"


def test_parse_handles_lowercase_arxiv_key() -> None:
    hits = _parse_s2_response(
        _s2_payload(
            [
                {
                    "title": "Foo",
                    "externalIds": {"arxiv": "1901.00002"},
                },
            ]
        )
    )
    assert hits and hits[0].arxiv_id == "1901.00002"


def test_parse_rejects_non_dict_payload() -> None:
    with pytest.raises(ArxivSearchError):
        _parse_s2_response(b"[1,2,3]")


def test_parse_handles_missing_data_field() -> None:
    with pytest.raises(ArxivSearchError):
        _parse_s2_response(json.dumps({"data": "not a list"}).encode("utf-8"))


# ----------------------------- HTTP behaviour ------------------------------


def _fake_resp(body: bytes) -> object:
    class FakeResp:
        def read(self) -> bytes:
            return body

        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    return FakeResp()


def test_search_end_to_end_returns_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _s2_payload(
        [
            {
                "title": "Attention Is All You Need",
                "externalIds": {"ArXiv": "1706.03762"},
                "year": 2017,
            }
        ]
    )
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.urllib.request.urlopen",
        lambda *a, **k: _fake_resp(payload),
    )
    hits = SemanticScholarSearcher(min_request_interval=0).search(
        "transformer", max_results=3
    )
    assert hits[0].arxiv_id == "1706.03762"
    assert hits[0].source == "semantic_scholar"


def test_search_includes_api_key_header_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    def fake_urlopen(req: urllib.request.Request, *_a: object, **_kw: object) -> object:
        captured["key"] = req.get_header("X-api-key")
        return _fake_resp(_s2_payload([]))

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.urllib.request.urlopen", fake_urlopen
    )
    SemanticScholarSearcher(min_request_interval=0, api_key="sk-test").search("x")
    assert captured["key"] == "sk-test"


def test_search_retries_on_http_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_urlopen(*_a: object, **_kw: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429)
        return _fake_resp(
            _s2_payload(
                [{"title": "ok", "externalIds": {"ArXiv": "1"}, "year": 2020}]
            )
        )

    sleeps: list[float] = []
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.time.sleep",
        lambda s: sleeps.append(s),
    )
    hits = SemanticScholarSearcher(
        retries=2, rate_limit_backoff=1.0, min_request_interval=0
    ).search("x")
    assert hits[0].arxiv_id == "1"
    assert sleeps == [1.0]


def test_search_429_final_message_mentions_api_key_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(429)),
    )
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.time.sleep", lambda _s: None
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        SemanticScholarSearcher(
            retries=1, rate_limit_backoff=0.0, min_request_interval=0
        ).search("x")
    assert "rate-limit" in str(exc_info.value).lower()
    assert "semanticscholar.org" in str(exc_info.value)


def test_search_timeout_wraps_into_arxiv_search_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: object, **_kw: object) -> object:
        raise TimeoutError("read timed out")

    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.urllib.request.urlopen", boom
    )
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.time.sleep", lambda _s: None
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        SemanticScholarSearcher(retries=0, min_request_interval=0).search("x")
    assert "Semantic Scholar timed out" in str(exc_info.value)


def test_search_wraps_json_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_agent.search.semantic_scholar.urllib.request.urlopen",
        lambda *a, **k: _fake_resp(b"not json"),
    )
    with pytest.raises(ArxivSearchError) as exc_info:
        SemanticScholarSearcher(min_request_interval=0).search("x")
    assert "unparseable" in str(exc_info.value)
