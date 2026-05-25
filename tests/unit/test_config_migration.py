"""Config migration: OpenRouter key/model vs legacy base_url."""

from __future__ import annotations

from pathlib import Path

import yaml

from research_agent.config import OPENROUTER_BASE_URL, Config


def test_load_migrates_deepseek_base_url_when_model_is_openrouter_slug(
    config_dir: Path,
) -> None:
    config_dir.mkdir(parents=True)
    path = config_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api_key": "sk-or-v1-test-key-12345",
                "model": "openai/gpt-4o-mini",
                "base_url": "https://api.deepseek.com",
                "data_dir": str(config_dir),
            }
        ),
        encoding="utf-8",
    )
    cfg = Config.load(config_dir)
    assert cfg.base_url == OPENROUTER_BASE_URL
    reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert reloaded["base_url"] == OPENROUTER_BASE_URL


def test_set_api_key_migrates_base_url(config_dir: Path) -> None:
    cfg = Config(
        data_dir=config_dir,
        api_key="old",
        model="openai/gpt-4o-mini",
        base_url="https://api.deepseek.com",
    )
    cfg.save()
    cfg.set_field("api_key", "sk-or-v1-new-key-99999")
    loaded = Config.load(config_dir)
    assert loaded.base_url == OPENROUTER_BASE_URL


def test_strips_whitespace_from_api_key(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir, api_key="  sk-or-v1-abc  ")
    assert cfg.api_key == "sk-or-v1-abc"
