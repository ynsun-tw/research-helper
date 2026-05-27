"""Base agent abstractions.

Structured LLM-output parsing moved to :mod:`research_agent.agents.schemas`
(Pydantic v2 models + ``parse_model``). The legacy ``extract_json`` helper
was removed once every agent migrated; if you need the raw envelope use
``schemas.strip_to_json``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from research_agent.agents.prompts import load_system_prompt
from research_agent.core.language import DEFAULT_LANGUAGE, response_language_instruction
from research_agent.core.llm import ChatMessage, LLMProvider


@dataclass
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
        prompt_stem: str | None = None,
    ) -> None:
        self.llm = llm
        self.language = language
        stem = prompt_stem or self.prompt_name
        base = system_prompt or load_system_prompt(stem)
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
