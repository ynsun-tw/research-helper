"""Tests for real provider-reported token telemetry (T3.4)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from research_agent.chat import run_chat
from research_agent.config import Config
from research_agent.core.llm import ChatResponse, MockLLMProvider, TokenUsage


def _cfg(d: Path) -> Config:
    return Config(data_dir=d, api_key="sk-x")


def test_token_usage_default_is_empty() -> None:
    u = TokenUsage()
    assert u.is_empty
    assert u.prompt_tokens == 0


def test_chat_response_carries_usage() -> None:
    resp = ChatResponse(
        content="ok",
        usage=TokenUsage(prompt_tokens=123, completion_tokens=45, total_tokens=168),
    )
    assert resp.usage is not None
    assert not resp.usage.is_empty
    assert resp.usage.prompt_tokens == 123


def test_token_footer_uses_real_usage_when_provided(
    config_dir: Path,
) -> None:
    """When the mock provider returns usage, the footer should show
    exact counts (no '~' prefix) instead of the heuristic."""
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120, force_terminal=False)

    canned = ChatResponse(
        content="Real number reply.",
        usage=TokenUsage(
            prompt_tokens=777, completion_tokens=42, total_tokens=819
        ),
    )
    llm = MockLLMProvider([canned])
    inputs = iter(["hi", "/exit"])

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _: next(inputs),
        use_chroma=False,
    )

    out = console.file.getvalue()
    # Footer should now use exact numbers — look for the real values
    # without the ``~`` prefix.
    assert "777 in" in out, f"expected real prompt count in output:\n{out}"
    assert "42 out" in out, f"expected real completion count in output:\n{out}"


def test_token_footer_falls_back_to_heuristic_without_usage(
    config_dir: Path,
) -> None:
    """When the provider doesn't supply usage, the footer should keep
    its ``~`` prefix to flag that the numbers are estimates."""
    cfg = _cfg(config_dir)
    console = Console(file=StringIO(), width=120, force_terminal=False)

    llm = MockLLMProvider(["plain text reply, no usage attached"])
    inputs = iter(["hi", "/exit"])

    run_chat(
        cfg,
        llm,
        console,
        input_fn=lambda _: next(inputs),
        use_chroma=False,
    )

    out = console.file.getvalue()
    assert "~" in out and "in" in out
