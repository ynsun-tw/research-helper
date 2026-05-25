"""Unit tests for configuration management."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from research_agent.config import CONFIG_FILE_MODE, Config, ConfigError


def test_load_returns_defaults_when_missing(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    assert cfg.api_key is None
    assert cfg.model == "deepseek-chat"
    assert cfg.data_dir == config_dir


def test_save_and_load_roundtrip(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-test-secret-key-12345")
    cfg.save()
    loaded = Config.load(config_dir)
    assert loaded.api_key == "sk-test-secret-key-12345"
    assert loaded.model == "deepseek-chat"


def test_config_file_mode_is_600(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-abc")
    cfg.save()
    mode = stat.S_IMODE(os.stat(cfg.config_path).st_mode)
    assert mode == CONFIG_FILE_MODE


def test_masked_api_key_not_set() -> None:
    cfg = Config()
    assert cfg.masked_api_key() == "(not set)"


def test_masked_api_key_redacts_long_key() -> None:
    cfg = Config(api_key="sk-abcdefghijklmnop")
    masked = cfg.masked_api_key()
    assert "abcdefghijklmnop" not in masked
    assert masked.startswith("sk-a")
    assert masked.endswith("mnop")


def test_masked_api_key_short_key() -> None:
    cfg = Config(api_key="short")
    assert cfg.masked_api_key() == "****"


def test_require_api_key_raises_when_missing() -> None:
    cfg = Config()
    with pytest.raises(ConfigError, match="API key is not configured"):
        cfg.require_api_key()


def test_require_api_key_returns_value() -> None:
    cfg = Config(api_key="sk-live")
    assert cfg.require_api_key() == "sk-live"


def test_set_field_unknown_key(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir)
    with pytest.raises(ConfigError, match="Unknown config key"):
        cfg.set_field("unknown", "x")


def test_set_field_persists(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir)
    cfg.set_field("api_key", "sk-xxx")
    cfg.set_field("model", "deepseek-reasoner")
    loaded = Config.load(config_dir)
    assert loaded.api_key == "sk-xxx"
    assert loaded.model == "deepseek-reasoner"


def test_get_field_masks_api_key(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-abcdefghijklmnop")
    assert "abcdefghijklmnop" not in cfg.get_field("api_key")


def test_invalid_yaml_raises(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    path = config_dir / "config.yaml"
    path.write_text("not-a-mapping", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid config format"):
        Config.load(config_dir)


def test_save_writes_valid_yaml(config_dir: Path) -> None:
    cfg = Config(data_dir=config_dir, api_key="sk-y", model="m1")
    cfg.save()
    with (config_dir / "config.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["api_key"] == "sk-y"
    assert data["model"] == "m1"
