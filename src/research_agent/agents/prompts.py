"""Load agent system prompts from packaged YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptNotFoundError(Exception):
    """Raised when a prompt file is missing or malformed."""


def load_system_prompt(name: str, *, prompts_dir: Path | None = None) -> str:
    """Load the ``system`` field from ``prompts/<name>.yaml``."""
    base = prompts_dir or PROMPTS_DIR
    path = base / f"{name}.yaml"
    if not path.exists():
        raise PromptNotFoundError(f"Prompt file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "system" not in data:
        raise PromptNotFoundError(f"Prompt {path} must contain a top-level 'system' key")
    system = data["system"]
    if not isinstance(system, str) or not system.strip():
        raise PromptNotFoundError(f"Prompt {path} has an empty 'system' field")
    return system.strip()
