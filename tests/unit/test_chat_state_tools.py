"""Phase 3 state-mutating chat tools.

train_style / build_fingerprint / update_fingerprint / get_config /
set_config all touch disk (samples DB, fingerprint JSON, config YAML).
Tests monkeypatch the heavy paths (PDF loading, fingerprint compute)
or use ``Config(data_dir=tmp_path)`` so each test gets a fresh
filesystem slot.

We do NOT test the prompt-only confirmation contract here — that's
LLM behaviour, validated by the C-stage human verification. The
tool layer is intentionally unguarded per ADR D3.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from research_agent.chat.session import ChatSession
from research_agent.chat.tools import (
    exec_build_fingerprint,
    exec_get_config,
    exec_set_config,
    exec_train_style,
    exec_update_fingerprint,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider


@pytest.fixture
def session(tmp_path) -> ChatSession:
    cfg = Config(data_dir=tmp_path, api_key="sk-or-test")
    console = Console(file=StringIO(), width=120)
    return ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=console,
        input_fn=lambda _: "",
        use_chroma=False,
    )


# ---- train_style -----------------------------------------------------------


def test_train_style_requires_input(session: ChatSession) -> None:
    out = exec_train_style(session, {})
    assert "Error" in out
    assert "at least one source" in out


def test_train_style_passes_sources_and_replace(
    session: ChatSession, monkeypatch
) -> None:
    from research_agent.cli_style import StyleTrainResult

    captured: dict[str, object] = {}

    def fake_run_style_train(cfg, console, **kwargs):
        captured.update(kwargs)
        return StyleTrainResult(
            sources_processed=2,
            sources_failed=0,
            paragraphs_added=24,
            paragraphs_skipped=0,
            per_source=[
                ("arxiv:2301.07041", 12, "ok"),
                ("arxiv:2305.14314", 12, "ok"),
            ],
        )

    monkeypatch.setattr("research_agent.cli_style.run_style_train", fake_run_style_train)
    out = exec_train_style(
        session,
        {"sources": ["arxiv:2301.07041", "arxiv:2305.14314"], "append": False},
    )
    assert captured["sources"] == ["arxiv:2301.07041", "arxiv:2305.14314"]
    assert captured["replace"] is True
    assert "24 paragraph" in out
    assert "build_fingerprint" in out


def test_train_style_append_flips_replace(
    session: ChatSession, monkeypatch
) -> None:
    from research_agent.cli_style import StyleTrainResult

    captured: dict[str, object] = {}

    def fake_run_style_train(cfg, console, **kwargs):
        captured.update(kwargs)
        return StyleTrainResult(1, 0, 5, 0, [("foo.pdf", 5, "ok")])

    monkeypatch.setattr("research_agent.cli_style.run_style_train", fake_run_style_train)
    exec_train_style(session, {"sources": ["arxiv:1"], "append": True})
    assert captured["replace"] is False


def test_train_style_surfaces_exception(
    session: ChatSession, monkeypatch
) -> None:
    def boom(cfg, console, **kwargs):
        raise RuntimeError("PDF parse failed")

    monkeypatch.setattr("research_agent.cli_style.run_style_train", boom)
    out = exec_train_style(session, {"sources": ["arxiv:x"]})
    assert "Error" in out
    assert "PDF parse failed" in out


# ---- build_fingerprint / update_fingerprint -------------------------------


def test_build_fingerprint_empty_corpus(session: ChatSession) -> None:
    out = exec_build_fingerprint(session, {})
    assert "Error" in out
    assert "train_style" in out


def test_build_fingerprint_success(session: ChatSession, monkeypatch) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.run_style_fingerprint",
        lambda cfg, console: 0,
    )
    out = exec_build_fingerprint(session, {})
    assert "Fingerprint built" in out
    assert str(session.cfg.fingerprint_path) in out


def test_update_fingerprint_empty_corpus(session: ChatSession) -> None:
    out = exec_update_fingerprint(session, {})
    assert "Error" in out
    assert "train_style" in out


def test_update_fingerprint_success(session: ChatSession, monkeypatch) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.run_style_update",
        lambda cfg, console: 0,
    )
    out = exec_update_fingerprint(session, {})
    assert "Fingerprint updated" in out
    assert "archived" in out


# ---- get_config -----------------------------------------------------------


def test_get_config_all_keys(session: ChatSession) -> None:
    out = exec_get_config(session, {})
    assert "model" in out
    assert "api_key" in out
    # api_key value must be masked, never the literal key.
    assert "sk-or-test" not in out
    assert "config_path" in out


def test_get_config_single_key_masks_api_key(session: ChatSession) -> None:
    out = exec_get_config(session, {"key": "api_key"})
    assert "api_key" in out
    assert "sk-or-test" not in out
    # masked format is "sk-o…test" (4 chars start + 4 chars end)
    assert "..." in out or "*" in out


def test_get_config_single_key_real_value(session: ChatSession) -> None:
    out = exec_get_config(session, {"key": "model"})
    assert "model =" in out
    # Default model from Config should be present (non-empty value).
    assert "=" in out


def test_get_config_unknown_key(session: ChatSession) -> None:
    out = exec_get_config(session, {"key": "no_such_field"})
    assert "Error" in out


# ---- set_config -----------------------------------------------------------


def test_set_config_writes_through(session: ChatSession, tmp_path) -> None:
    out = exec_set_config(session, {"key": "model", "value": "deepseek/test"})
    assert "Set model" in out
    assert session.cfg.model == "deepseek/test"
    reloaded = Config.load(data_dir=tmp_path)
    assert reloaded.model == "deepseek/test"


def test_set_config_masks_api_key(session: ChatSession) -> None:
    out = exec_set_config(
        session, {"key": "api_key", "value": "sk-or-very-secret-123"}
    )
    assert "Set api_key" in out
    assert "sk-or-very-secret-123" not in out


def test_set_config_unknown_key(session: ChatSession) -> None:
    out = exec_set_config(session, {"key": "no_such_field", "value": "x"})
    assert "Error" in out


def test_set_config_missing_value(session: ChatSession) -> None:
    out = exec_set_config(session, {"key": "model"})
    assert "value is required" in out


def test_set_config_validates_alert_threshold(session: ChatSession) -> None:
    out = exec_set_config(
        session, {"key": "alert_threshold", "value": "not-a-number"}
    )
    assert "Error" in out
    assert "alert_threshold" in out


# ---- registry --------------------------------------------------------------


def test_phase3_tools_registered() -> None:
    from research_agent.chat.tools import LLM_TOOLS

    for name in (
        "train_style",
        "build_fingerprint",
        "update_fingerprint",
        "get_config",
        "set_config",
    ):
        assert name in LLM_TOOLS, f"missing tool: {name}"
