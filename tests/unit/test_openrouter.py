"""OpenRouter-specific LLM client behaviour."""

from __future__ import annotations

from research_agent.config import OPENROUTER_BASE_URL, Config
from research_agent.core.llm import LLMClient, _openrouter_headers


def test_openrouter_headers_when_base_url_matches(config_dir) -> None:
    cfg = Config(
        data_dir=config_dir,
        api_key="sk-or-test",
        base_url=OPENROUTER_BASE_URL,
        app_title="My App",
        app_url="https://example.com",
    )
    headers = _openrouter_headers(cfg)
    assert headers == {"HTTP-Referer": "https://example.com", "X-Title": "My App"}


def test_no_extra_headers_for_custom_base_url(config_dir) -> None:
    cfg = Config(data_dir=config_dir, base_url="https://api.deepseek.com")
    assert _openrouter_headers(cfg) is None


def test_llm_client_from_config_sets_openrouter_headers(config_dir) -> None:
    cfg = Config(
        data_dir=config_dir,
        api_key="sk-or-test",
        base_url=OPENROUTER_BASE_URL,
    )
    client = LLMClient.from_config(cfg)
    assert client._client.default_headers is not None
    assert client._client.default_headers.get("X-Title") == "Research Agent"
