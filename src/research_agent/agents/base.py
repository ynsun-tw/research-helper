"""Base agent abstractions."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.prompts import load_system_prompt
from research_agent.core.language import DEFAULT_LANGUAGE, response_language_instruction
from research_agent.core.llm import ChatMessage, LLMProvider


@dataclass(slots=True)
class AgentResponse:
    """Standard agent output envelope."""

    role: str
    content: str
    confidence: float | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """All agents share an LLM backend and a YAML system prompt."""

    role: str = "agent"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        system_prompt: str | None = None,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self.llm = llm
        self.language = language
        base = system_prompt or load_system_prompt(self.prompt_name)
        self.system_prompt = f"{base}\n\n{response_language_instruction(language)}"

    @property
    @abstractmethod
    def prompt_name(self) -> str:
        """YAML stem under ``prompts/`` (e.g. ``analyst``)."""

    @abstractmethod
    def run(self, context: dict[str, Any]) -> AgentResponse:
        """Execute the agent for the given context dict."""

    def _chat(self, user_content: str, *, temperature: float = 0.4) -> str:
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        return self.llm.chat(messages, temperature=temperature)


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM reply, tolerating optional markdown fences."""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1:
            stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON from model output: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from model output")
    return data
