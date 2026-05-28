"""Configuration management for Research Agent."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from research_agent.core.language import DEFAULT_LANGUAGE, normalize_language

DEFAULT_DATA_DIR = Path.home() / ".research-agent"
CONFIG_FILENAME = "config.yaml"
CONFIG_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 600

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-chat"

KNOWN_KEYS = frozenset(
    {
        "api_key",
        "model",
        "base_url",
        "data_dir",
        "app_title",
        "app_url",
        "language",
        "alert_threshold",
        "context_window_tokens",
        "reserve_tokens_for_output",
    }
)

DEFAULT_ALERT_THRESHOLD = 0.8
# 0 means "auto-detect from model slug" via memory.lookup_model_context_window.
DEFAULT_CONTEXT_WINDOW_TOKENS = 0
DEFAULT_RESERVE_TOKENS_FOR_OUTPUT = 4000


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


def _is_openrouter_key(api_key: str | None) -> bool:
    return bool(api_key and api_key.startswith("sk-or-"))


def _is_openrouter_model(model: str) -> bool:
    """OpenRouter model slugs use ``provider/model`` form."""
    return "/" in model


def should_use_openrouter(cfg: Config) -> bool:
    return (
        "openrouter.ai" in cfg.base_url
        or _is_openrouter_key(cfg.api_key)
        or _is_openrouter_model(cfg.model)
    )


class Config(BaseModel):
    """Application configuration persisted to ~/.research-agent/config.yaml."""

    api_key: str | None = None
    model: str = OPENROUTER_DEFAULT_MODEL
    base_url: str = OPENROUTER_BASE_URL
    app_title: str = "Research Agent"
    app_url: str = "https://github.com/research-agent"
    language: str = DEFAULT_LANGUAGE
    data_dir: Path = Field(default_factory=lambda: DEFAULT_DATA_DIR)
    # Cosine similarity threshold for "this paper is related to a parked
    # idea you might want to revisit" banners. M3 S3.4.1 spec: 0.8.
    # Values must lie in [0.0, 1.0]; out-of-range YAML values are
    # clamped silently rather than crashing config loading.
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD
    # T2.2 context-budget knobs. ``0`` = auto-detect from the model slug
    # via ``research_agent.memory.lookup_model_context_window``.
    # ``reserve_tokens_for_output`` is subtracted from the detected window
    # so the reply has room to land. Both default to safe values that
    # preserve the pre-T2.2 fixed-8K behaviour on unknown models.
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    reserve_tokens_for_output: int = DEFAULT_RESERVE_TOKENS_FOR_OUTPUT

    @field_validator("api_key", mode="before")
    @classmethod
    def _strip_api_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("base_url", "model", "app_title", "app_url", mode="before")
    @classmethod
    def _strip_str_fields(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, value: Any) -> str:
        return normalize_language(value)

    @field_validator("data_dir", mode="before")
    @classmethod
    def _coerce_data_dir(cls, value: Any) -> Path:
        return Path(value) if value is not None else DEFAULT_DATA_DIR

    @field_validator("alert_threshold", mode="before")
    @classmethod
    def _clamp_alert_threshold(cls, value: Any) -> float:
        if value is None:
            return DEFAULT_ALERT_THRESHOLD
        try:
            v = float(value)
        except (TypeError, ValueError):
            return DEFAULT_ALERT_THRESHOLD
        return max(0.0, min(1.0, v))

    @field_validator(
        "context_window_tokens", "reserve_tokens_for_output", mode="before"
    )
    @classmethod
    def _coerce_token_budget(cls, value: Any) -> int:
        if value is None:
            return 0
        try:
            v = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, v)

    def ensure_openrouter_alignment(self) -> Config:
        """Fix base_url when key/model clearly target OpenRouter but URL does not."""
        if not should_use_openrouter(self):
            return self
        if "openrouter.ai" in self.base_url:
            return self
        return self.model_copy(update={"base_url": OPENROUTER_BASE_URL})

    @property
    def config_path(self) -> Path:
        return self.data_dir / CONFIG_FILENAME

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def pdf_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "papers"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def style_dir(self) -> Path:
        """Directory for style fingerprint JSON + (future) version history."""
        return self.data_dir / "style"

    @property
    def fingerprint_path(self) -> Path:
        return self.style_dir / "fingerprint.json"

    @classmethod
    def load(cls, data_dir: Path | None = None) -> Config:
        """Load config from disk, or return defaults if the file does not exist."""
        base = data_dir or DEFAULT_DATA_DIR
        path = base / CONFIG_FILENAME
        if not path.exists():
            return cls(data_dir=base)
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"Invalid config format in {path}")
        raw.setdefault("data_dir", str(base))
        cfg = cls.model_validate(raw).ensure_openrouter_alignment()
        if cfg.base_url != raw.get("base_url") and "openrouter.ai" in cfg.base_url:
            cfg.save()
        return cfg

    def save(self) -> None:
        """Persist config to YAML with file mode 600."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "app_title": self.app_title,
            "app_url": self.app_url,
            "language": self.language,
            "data_dir": str(self.data_dir),
            "alert_threshold": self.alert_threshold,
            "context_window_tokens": self.context_window_tokens,
            "reserve_tokens_for_output": self.reserve_tokens_for_output,
        }
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, allow_unicode=True)
        os.chmod(self.config_path, CONFIG_FILE_MODE)

    def masked_api_key(self) -> str:
        """Return a redacted representation of the API key for display."""
        if not self.api_key:
            return "(not set)"
        key = self.api_key
        if len(key) <= 8:
            return "****"
        return f"{key[:4]}...{key[-4:]}"

    def require_api_key(self) -> str:
        """Return the API key or raise a friendly ConfigError."""
        if not self.api_key:
            raise ConfigError(
                "API key is not configured. Get a key at https://openrouter.ai/keys "
                "then run: research config set api_key <your-key>"
            )
        return self.api_key

    def set_field(self, key: str, value: str) -> None:
        """Update a single config field and save."""
        if key not in KNOWN_KEYS:
            raise ConfigError(
                f"Unknown config key '{key}'. Valid keys: {', '.join(sorted(KNOWN_KEYS))}"
            )
        try:
            if key == "data_dir":
                setattr(self, key, Path(value))
            elif key == "language":
                setattr(self, key, normalize_language(value))
            elif key == "alert_threshold":
                try:
                    fv = float(value)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        "alert_threshold must be a number in [0.0, 1.0] "
                        "(e.g. 0.8)"
                    ) from exc
                if not 0.0 <= fv <= 1.0:
                    raise ConfigError(
                        f"alert_threshold must be in [0.0, 1.0], got {fv}"
                    )
                setattr(self, key, fv)
            elif key in {"context_window_tokens", "reserve_tokens_for_output"}:
                try:
                    iv = int(value)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"{key} must be a non-negative integer "
                        "(0 = auto-detect for context_window_tokens)"
                    ) from exc
                if iv < 0:
                    raise ConfigError(
                        f"{key} must be >= 0, got {iv}"
                    )
                setattr(self, key, iv)
            else:
                setattr(self, key, value.strip() if isinstance(value, str) else value)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        aligned = self.ensure_openrouter_alignment()
        if aligned.base_url != self.base_url:
            self.base_url = aligned.base_url
        self.save()

    def get_field(self, key: str) -> str:
        """Get a config field value (api_key is masked)."""
        if key not in KNOWN_KEYS:
            raise ConfigError(
                f"Unknown config key '{key}'. Valid keys: {', '.join(sorted(KNOWN_KEYS))}"
            )
        if key == "api_key":
            return self.masked_api_key()
        value = getattr(self, key)
        return str(value)
