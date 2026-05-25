"""Tests for response language configuration."""

from __future__ import annotations

import pytest

from research_agent.agents.analyst import Analyst
from research_agent.config import Config, ConfigError
from research_agent.core.language import (
    DEFAULT_LANGUAGE,
    normalize_language,
    response_language_instruction,
)
from research_agent.core.llm import MockLLMProvider


def test_normalize_language_defaults() -> None:
    assert normalize_language(None) == "en"
    assert normalize_language("english") == "en"
    assert normalize_language("中文") == "zh"
    assert normalize_language("ZH") == "zh"


def test_normalize_language_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_language("fr")


def test_config_default_language(config_dir) -> None:
    cfg = Config.load(config_dir)
    assert cfg.language == "en"


def test_config_set_language_zh(config_dir) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-x")
    cfg.set_field("language", "chinese")
    loaded = Config.load(config_dir)
    assert loaded.language == "zh"


def test_config_set_invalid_language(config_dir) -> None:
    cfg = Config(data_dir=config_dir)
    with pytest.raises(ConfigError, match="Unsupported"):
        cfg.set_field("language", "klingon")


def test_analyst_system_prompt_includes_language_instruction() -> None:
    llm = MockLLMProvider()
    agent_en = Analyst(llm, language="en")
    agent_zh = Analyst(llm, language="zh")
    assert "English" in agent_en.system_prompt
    assert "简体中文" in agent_zh.system_prompt


def test_response_language_instruction_zh() -> None:
    assert "简体中文" in response_language_instruction("zh")
    assert DEFAULT_LANGUAGE == "en"
