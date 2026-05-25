"""LLM client with provider abstraction, streaming, and retries."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionChunk

from research_agent.config import OPENROUTER_BASE_URL, Config


class LLMError(Exception):
    """User-facing LLM failure (no stack trace in CLI)."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class LLMProvider(ABC):
    """Abstract LLM backend for swapping models or injecting mocks in tests."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Return the full assistant reply."""

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield assistant reply tokens/chunks."""


class LLMClient(LLMProvider):
    """OpenAI-compatible client (default: OpenRouter)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        extra_headers: Mapping[str, str] | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.base_delay = base_delay
        headers = dict(extra_headers or {})
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=headers or None,
        )

    @classmethod
    def from_config(cls, config: Config) -> LLMClient:
        headers = _openrouter_headers(config)
        return cls(
            api_key=config.require_api_key(),
            model=config.model,
            base_url=config.base_url,
            extra_headers=headers,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        chunks = list(self.chat_stream(messages, temperature=temperature, max_tokens=max_tokens))
        return "".join(chunks)

    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = _to_openai_messages(messages)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=cast(Any, payload),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    if not isinstance(chunk, ChatCompletionChunk):
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except (APIConnectionError, RateLimitError, APIStatusError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.base_delay * (2**attempt))

        raise LLMError(_friendly_message(last_error, base_url=self.base_url)) from last_error


class MockLLMProvider(LLMProvider):
    """Deterministic LLM for unit/integration tests."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[ChatMessage]] = []

    def enqueue(self, response: str) -> None:
        self._responses.append(response)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        return "".join(self.chat_stream(messages, temperature=temperature, max_tokens=max_tokens))

    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self.calls.append(list(messages))
        if not self._responses:
            yield '{"error": "no mock response queued"}'
            return
        text = self._responses.pop(0)
        yield text


def _openrouter_headers(config: Config) -> dict[str, str] | None:
    """OpenRouter recommends HTTP-Referer and X-Title on each request."""
    if "openrouter.ai" not in config.base_url:
        return None
    return {
        "HTTP-Referer": config.app_url,
        "X-Title": config.app_title,
    }


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _friendly_message(exc: Exception | None, *, base_url: str = "") -> str:
    if exc is None:
        return "The language model request failed. Please try again later."
    if isinstance(exc, APIConnectionError):
        return "Cannot reach the LLM API. Check your network connection and try again."
    if isinstance(exc, RateLimitError):
        return "LLM API rate limit exceeded. Wait a moment and try again."
    if isinstance(exc, APIStatusError):
        detail = _extract_api_error_detail(exc)
        if exc.status_code == 401:
            hint = (
                "Authentication failed (401). "
                f"Request was sent to: {base_url or '(unknown)'}. "
            )
            if base_url and "openrouter.ai" not in base_url:
                hint += (
                    "Your model/key look like OpenRouter — run: "
                    "research config set base_url https://openrouter.ai/api/v1 "
                )
            else:
                hint += (
                    "Check your OpenRouter key at https://openrouter.ai/keys — "
                    "then: research config set api_key <your-key> "
                )
            if detail:
                hint += f"API says: {detail}"
            return hint.strip()
        if exc.status_code == 429:
            return "LLM API rate limit exceeded. Wait a moment and try again."
        msg = f"LLM API error ({exc.status_code})"
        if base_url:
            msg += f" at {base_url}"
        if detail:
            msg += f": {detail}"
        return msg
    return f"LLM request failed: {exc}"


def _extract_api_error_detail(exc: APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
    if body:
        return str(body)[:200]
    return str(exc.message)[:200] if getattr(exc, "message", None) else ""
