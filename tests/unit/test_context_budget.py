"""Tests for model-aware context budget (T2.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.config import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_RESERVE_TOKENS_FOR_OUTPUT,
    Config,
    ConfigError,
)
from research_agent.memory import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    lookup_model_context_window,
    resolve_context_budget,
)


def test_lookup_known_models() -> None:
    assert lookup_model_context_window("deepseek/deepseek-chat") == 64_000
    assert lookup_model_context_window("anthropic/claude-3-5-sonnet") == 200_000
    assert lookup_model_context_window("openai/gpt-4o") == 128_000


def test_lookup_unknown_returns_default() -> None:
    assert (
        lookup_model_context_window("random/proprietary-model")
        == DEFAULT_MAX_CONTEXT_TOKENS
    )
    assert lookup_model_context_window("") == DEFAULT_MAX_CONTEXT_TOKENS


def test_lookup_is_case_insensitive() -> None:
    assert (
        lookup_model_context_window("DeepSeek/DeepSeek-Chat")
        == lookup_model_context_window("deepseek/deepseek-chat")
    )


def test_resolve_budget_subtracts_reserve_for_output() -> None:
    # DeepSeek = 64K, reserve = 4K → 60K
    assert (
        resolve_context_budget(
            "deepseek/deepseek-chat", reserve_output=4000
        )
        == 60_000
    )


def test_resolve_budget_respects_override() -> None:
    assert (
        resolve_context_budget(
            "deepseek/deepseek-chat", override=12_000, reserve_output=1000
        )
        == 11_000
    )


def test_resolve_budget_never_goes_below_legacy_default() -> None:
    """The 8K floor preserves pre-T2.2 behaviour for unknown small models."""
    assert (
        resolve_context_budget("tiny-experimental-model", reserve_output=4000)
        == DEFAULT_MAX_CONTEXT_TOKENS
    )


def test_config_defaults(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    assert cfg.context_window_tokens == DEFAULT_CONTEXT_WINDOW_TOKENS
    assert cfg.reserve_tokens_for_output == DEFAULT_RESERVE_TOKENS_FOR_OUTPUT


def test_config_persists_and_loads_token_fields(tmp_path: Path) -> None:
    cfg = Config(
        data_dir=tmp_path,
        api_key="sk-x",
        context_window_tokens=32_000,
        reserve_tokens_for_output=2_000,
    )
    cfg.save()
    reloaded = Config.load(data_dir=tmp_path)
    assert reloaded.context_window_tokens == 32_000
    assert reloaded.reserve_tokens_for_output == 2_000


def test_config_set_field_validates_token_keys(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path, api_key="sk-x")
    cfg.set_field("context_window_tokens", "20000")
    assert cfg.context_window_tokens == 20_000

    with pytest.raises(ConfigError):
        cfg.set_field("context_window_tokens", "not-a-number")
    with pytest.raises(ConfigError):
        cfg.set_field("reserve_tokens_for_output", "-5")


def test_estimate_tokens_uses_tiktoken_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When tiktoken is present, estimate_tokens should defer to it."""
    from research_agent.memory import working_memory as wm

    calls: list[str] = []

    class FakeEncoder:
        def encode(self, text: str) -> list[int]:
            calls.append(text)
            # Pretend we tokenized into one token per word.
            return text.split()

    monkeypatch.setattr(wm, "_ENCODER", FakeEncoder())
    assert wm.estimate_tokens("hello world this is a test") == 6
    assert calls == ["hello world this is a test"]


def test_estimate_tokens_falls_back_when_tiktoken_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_agent.memory import working_memory as wm

    monkeypatch.setattr(wm, "_ENCODER", None)
    text = "x" * 40  # 40 / 4 = 10
    assert wm.estimate_tokens(text) == 10


def test_estimate_tokens_swallows_encoder_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If tiktoken raises on a weird input, we must still return a number
    (the fallback heuristic) — never crash the prompt-builder."""
    from research_agent.memory import working_memory as wm

    class BadEncoder:
        def encode(self, text: str) -> list[int]:
            raise RuntimeError("boom")

    monkeypatch.setattr(wm, "_ENCODER", BadEncoder())
    text = "x" * 40
    assert wm.estimate_tokens(text) == 10


def test_chat_session_uses_model_aware_budget(
    config_dir: Path,
) -> None:
    """ChatSession.create should pick up the model-aware budget instead
    of the old hardcoded 8000."""
    from io import StringIO

    from rich.console import Console

    from research_agent.chat.session import ChatSession
    from research_agent.core.llm import MockLLMProvider

    cfg = Config(
        data_dir=config_dir,
        api_key="sk-x",
        model="anthropic/claude-3-5-sonnet",
    )
    session = ChatSession.create(
        cfg=cfg,
        llm=MockLLMProvider([]),
        console=Console(file=StringIO(), width=120),
        input_fn=lambda _: "",
        use_chroma=False,
    )
    # Claude 200K minus 4K reserve = 196K, far above the old 8K floor.
    assert session.max_context_tokens > 100_000
