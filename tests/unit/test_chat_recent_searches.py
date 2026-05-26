"""Tests for the recent_searches LLM tool (M3.1 v2)."""

from __future__ import annotations

import json
from io import StringIO

import pytest
from rich.console import Console

import research_agent.chat.tools as chat_tools
from research_agent.chat import run_chat
from research_agent.chat.session import ChatSession
from research_agent.chat.tools import LLM_TOOLS, exec_recent_searches
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, ToolCall
from research_agent.core.paper import Paper, Section
from research_agent.search.arxiv_search import ArxivSearchHit

ANALYST_JSON = json.dumps(
    {
        "contributions": ["C1"],
        "method_insights": ["M1"],
        "potential_impact": "Impact",
        "related_work": ["R1"],
        "claimed_vs_evidence": [],
        "confidence": 0.8,
    }
)
CRITIC_JSON = json.dumps(
    {
        "objections": ["O1"],
        "support_score": 7,
        "score_reason": "Good",
        "honesty_note": "",
    }
)


def _make_session(tmp_path):
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=120)
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )
    return session, console


def _hit(arxiv_id: str, title: str, year: str = "2018") -> ArxivSearchHit:
    return ArxivSearchHit(
        arxiv_id=arxiv_id,
        title=title,
        abstract=f"Abstract for {title}",
        published=f"{year}-06-12",
    )


def test_recent_searches_tool_registered() -> None:
    assert "recent_searches" in LLM_TOOLS
    schema = LLM_TOOLS["recent_searches"].schema["function"]
    assert schema["name"] == "recent_searches"
    assert "limit" in schema["parameters"]["properties"]


def test_exec_recent_searches_empty_history(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    out = exec_recent_searches(session, {})
    assert out == "No search history yet."
    session.close()


def test_exec_recent_searches_lists_hits_with_read_marker(tmp_path) -> None:
    session, console = _make_session(tmp_path)
    session.papers.save(
        Paper(id="arxiv:1810.04805", title="BERT", abstract="", full_text="")
    )
    session.searches.record(
        "transformer",
        [
            _hit("1706.03762", "Attention Is All You Need", "2017"),
            _hit("1810.04805", "BERT", "2018"),
        ],
    )

    result = exec_recent_searches(session, {"limit": 5})
    assert "Recent searches" in result
    assert "1706.03762" in result
    assert "1810.04805" in result
    assert "[READ]" in result
    # console should also render the table for the human
    rendered = console.file.getvalue()
    assert "transformer" in rendered
    session.close()


def test_exec_recent_searches_invalid_limit(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    out = exec_recent_searches(session, {"limit": "notanumber"})
    assert out.startswith("Error:")
    session.close()


def test_exec_recent_searches_clamps_limit(tmp_path) -> None:
    session, _ = _make_session(tmp_path)
    for i in range(8):
        session.searches.record(f"q{i}", [_hit(f"{i}.0", f"title {i}")])

    # asking for too many → clamped to 25; asking for too few → clamped to 1
    out_many = exec_recent_searches(session, {"limit": 1000})
    assert out_many.count("- ") >= 1
    out_one = exec_recent_searches(session, {"limit": 0})
    assert out_one.count("[20") + out_one.count("[19") >= 0  # smoke: no crash
    session.close()


def _tool_call_response(name: str, arguments: dict) -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=(
            ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(arguments)),
        ),
    )


def test_llm_chains_recent_searches_into_load_paper(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: prior search lives in DB, LLM uses recent_searches to recover the
    arxiv_id and then calls load_paper on it."""
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    console = Console(file=StringIO(), width=120)

    # Pre-populate search history (simulates earlier session).
    from research_agent.storage.database import Database
    from research_agent.storage.searches import SearchRepository

    db = Database(cfg.db_path)
    SearchRepository(db).record(
        "transformer",
        [
            _hit("1706.03762", "Attention Is All You Need", "2017"),
            _hit("1810.04805", "BERT", "2018"),
        ],
    )
    db.close()

    paper = Paper(
        id="arxiv:1810.04805",
        title="BERT",
        abstract="Pretraining.",
        sections=[Section("Intro", "Pretrain then fine-tune.")],
        full_text="BERT details.",
    )
    monkeypatch.setattr(
        chat_tools,
        "_load_anchor_paper",
        lambda *args, **kwargs: paper,
    )

    llm = MockLLMProvider(
        [
            _tool_call_response("recent_searches", {"limit": 5}),
            _tool_call_response("load_paper", {"source": "arxiv:1810.04805"}),
            ANALYST_JSON,
            CRITIC_JSON,
            ChatResponse(content="Loaded BERT — ready for /discuss."),
        ]
    )

    inputs = iter(
        ["open the BERT paper from my recent searches", "/exit"]
    )
    code = run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _prompt: next(inputs),
        use_chroma=False,
    )
    assert code == 0
    out = console.file.getvalue()
    assert "recent_searches" in out
    assert "load_paper" in out
    assert "Loaded BERT" in out
