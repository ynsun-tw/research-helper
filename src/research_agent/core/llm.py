"""LLM client with provider abstraction, streaming, retries, and tool calling."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionChunk

from research_agent.config import OPENROUTER_BASE_URL, Config


class LLMError(Exception):
    """User-facing LLM failure (no stack trace in CLI)."""


@dataclass(frozen=True)
class ToolCall:
    """One function-call request emitted by the LLM (OpenAI-compatible)."""

    id: str
    name: str
    arguments: str  # JSON-encoded; parse with ``json.loads`` on the executor side


@dataclass(frozen=True)
class ChatMessage:
    """One conversation message, optionally carrying tool-call metadata."""

    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    name: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token counts (T3.4).

    Mirrors the ``usage`` block returned by OpenAI-compatible APIs.
    When the provider doesn't supply usage (e.g. local mocks, partial
    streams), all three counts stay at ``0`` and consumers fall back
    to the in-process ``estimate_tokens`` heuristic.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return self.prompt_tokens == 0 and self.completion_tokens == 0


@dataclass(frozen=True)
class ChatResponse:
    """Structured LLM reply: text plus any requested tool calls."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


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
        """Return the full assistant reply (text only, no tool-call dispatch)."""

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield assistant reply tokens/chunks."""

    def chat_with_tools(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Return ChatResponse with text + optional tool_calls.

        Default falls back to plain ``chat`` (no tool support).
        Subclasses that support function-calling should override this.
        """
        text = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return ChatResponse(content=text)

    def chat_with_tools_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        on_content_delta: Callable[[str], None] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Streaming variant: call ``on_content_delta(chunk)`` per text chunk.

        Returns the same ``ChatResponse`` as ``chat_with_tools`` once the stream
        completes (full content + accumulated tool_calls).

        Default implementation does a single non-streaming call and emits the
        whole content as one callback — backends without real streaming (mocks,
        old providers) keep working with the same call site.
        """
        resp = self.chat_with_tools(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if on_content_delta and resp.content:
            on_content_delta(resp.content)
        return resp


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

    def chat_with_tools(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, payload),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                tcs: list[ToolCall] = []
                for tc in msg.tool_calls or []:
                    tcs.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments or "{}",
                        )
                    )
                return ChatResponse(
                    content=msg.content or "",
                    tool_calls=tuple(tcs),
                    usage=_extract_usage(getattr(resp, "usage", None)),
                )
            except (APIConnectionError, RateLimitError, APIStatusError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.base_delay * (2**attempt))
        raise LLMError(_friendly_message(last_error, base_url=self.base_url)) from last_error

    def chat_with_tools_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        on_content_delta: Callable[[str], None] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, payload),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # T3.4: ask OpenAI/OpenRouter for the usage block. It arrives
            # on the final chunk (with empty ``choices``) and is the only
            # way to get real token counts during a streaming completion.
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                stream = self._client.chat.completions.create(**kwargs)
                content_parts: list[str] = []
                # OpenAI streams tool_calls as indexed partial deltas:
                # each chunk may set ``id``/``function.name`` once and append
                # to ``function.arguments``. We accumulate into a dict keyed
                # by stream index and finalize at the end.
                tc_acc: dict[int, dict[str, str]] = {}
                usage_payload: Any = None
                for chunk in stream:
                    if not isinstance(chunk, ChatCompletionChunk):
                        continue
                    # Final usage chunk has empty choices but a populated
                    # ``usage`` block; capture it before bailing on choices.
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        usage_payload = chunk_usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        content_parts.append(text)
                        if on_content_delta is not None:
                            on_content_delta(text)
                    raw_tcs = getattr(delta, "tool_calls", None) or []
                    for tc in raw_tcs:
                        idx = tc.index if tc.index is not None else 0
                        entry = tc_acc.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            entry["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                entry["name"] = fn.name or ""
                            args = getattr(fn, "arguments", None)
                            if args:
                                entry["arguments"] += args
                tcs: list[ToolCall] = []
                for idx in sorted(tc_acc):
                    entry = tc_acc[idx]
                    if not entry["name"]:
                        continue
                    tcs.append(
                        ToolCall(
                            id=entry["id"] or f"call_{uuid.uuid4().hex[:8]}",
                            name=entry["name"],
                            arguments=entry["arguments"] or "{}",
                        )
                    )
                return ChatResponse(
                    content="".join(content_parts),
                    tool_calls=tuple(tcs),
                    usage=_extract_usage(usage_payload),
                )
            except (APIConnectionError, RateLimitError, APIStatusError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.base_delay * (2**attempt))
        raise LLMError(_friendly_message(last_error, base_url=self.base_url)) from last_error


class MockLLMProvider(LLMProvider):
    """Deterministic LLM for unit/integration tests.

    Accepts a list of canned responses. Each item may be a string (text reply)
    or a ``ChatResponse`` (text + optional tool_calls). ``chat`` returns text
    only; ``chat_with_tools`` returns the full ``ChatResponse``.
    """

    def __init__(self, responses: list[str | ChatResponse] | None = None) -> None:
        self._responses: list[ChatResponse] = []
        for r in responses or []:
            self._responses.append(_normalize_response(r))
        self.calls: list[list[ChatMessage]] = []
        self.tool_calls_seen: list[Sequence[dict[str, Any]] | None] = []

    def enqueue(self, response: str | ChatResponse) -> None:
        self._responses.append(_normalize_response(response))

    def enqueue_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        import json

        call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=json.dumps(arguments),
        )
        self._responses.append(ChatResponse(content="", tool_calls=(call,)))

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
        self.tool_calls_seen.append(None)
        if not self._responses:
            yield '{"error": "no mock response queued"}'
            return
        yield self._responses.pop(0).content

    def chat_with_tools(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        self.tool_calls_seen.append(list(tools) if tools else None)
        if not self._responses:
            return ChatResponse(content='{"error": "no mock response queued"}')
        return self._responses.pop(0)

    def chat_with_tools_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        on_content_delta: Callable[[str], None] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        resp = self.chat_with_tools(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if on_content_delta and resp.content:
            on_content_delta(resp.content)
        return resp


def _normalize_response(r: str | ChatResponse) -> ChatResponse:
    if isinstance(r, ChatResponse):
        return r
    return ChatResponse(content=r)


def _extract_usage(raw: Any) -> TokenUsage | None:
    """Best-effort coercion of an OpenAI ``CompletionUsage`` into TokenUsage.

    Returns ``None`` on missing input so the downstream token-footer
    code can fall back to its heuristic without conditionally branching
    on attribute access.
    """
    if raw is None:
        return None
    try:
        return TokenUsage(
            prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(raw, "total_tokens", 0) or 0),
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _openrouter_headers(config: Config) -> dict[str, str] | None:
    """OpenRouter recommends HTTP-Referer and X-Title on each request."""
    if "openrouter.ai" not in config.base_url:
        return None
    return {
        "HTTP-Referer": config.app_url,
        "X-Title": config.app_title,
    }


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.role}
        # OpenAI requires "content" key even for tool/assistant-with-tool-calls;
        # use null only when there are tool_calls and no text content.
        if m.tool_calls and not m.content:
            entry["content"] = None
        else:
            entry["content"] = m.content
        if m.tool_call_id is not None:
            entry["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            entry["name"] = m.name
        if m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in m.tool_calls
            ]
        out.append(entry)
    return out


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
