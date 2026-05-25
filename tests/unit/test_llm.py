"""Unit tests for LLM client and mock provider."""

from __future__ import annotations

from unittest.mock import MagicMock

from openai import APIConnectionError, APIStatusError

from research_agent.config import Config
from research_agent.core.llm import (
    ChatMessage,
    LLMClient,
    MockLLMProvider,
    _friendly_message,
)


def test_mock_llm_returns_queued_responses() -> None:
    mock = MockLLMProvider(["hello", "world"])
    assert mock.chat([ChatMessage("user", "hi")]) == "hello"
    assert mock.chat([ChatMessage("user", "again")]) == "world"
    assert len(mock.calls) == 2


def test_mock_llm_stream_yields_full_text() -> None:
    mock = MockLLMProvider(['{"ok": true}'])
    chunks = list(mock.chat_stream([ChatMessage("user", "go")]))
    assert "".join(chunks) == '{"ok": true}'


def test_llm_client_from_config(config_dir) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test")
    client = LLMClient.from_config(cfg)
    assert client.model == "deepseek/deepseek-chat"


def test_friendly_message_connection() -> None:
    exc = APIConnectionError(request=MagicMock())
    assert "network" in _friendly_message(exc).lower()


def test_friendly_message_unauthorized() -> None:
    exc = APIStatusError(
        "unauthorized",
        response=MagicMock(status_code=401),
        body=None,
    )
    msg = _friendly_message(exc, base_url="https://api.deepseek.com")
    assert "401" in msg
    assert "openrouter.ai/api/v1" in msg


def test_friendly_message_unauthorized_openrouter() -> None:
    exc = APIStatusError(
        "unauthorized",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid credentials"}},
    )
    msg = _friendly_message(exc, base_url="https://openrouter.ai/api/v1")
    assert "openrouter.ai" in msg
    assert "Invalid credentials" in msg


def test_friendly_message_none() -> None:
    assert "failed" in _friendly_message(None).lower()
