"""User-facing response language configuration."""

from __future__ import annotations

SUPPORTED_LANGUAGES = frozenset({"en", "zh"})
DEFAULT_LANGUAGE = "en"

_LANGUAGE_ALIASES: dict[str, str] = {
    "en": "en",
    "english": "en",
    "英文": "en",
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
    "中文": "zh",
    "简体中文": "zh",
}


def normalize_language(value: object) -> str:
    """Normalize CLI/config input to ``en`` or ``zh``."""
    if value is None:
        return DEFAULT_LANGUAGE
    key = str(value).strip().lower()
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    if key in SUPPORTED_LANGUAGES:
        return key
    supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
    raise ValueError(
        f"Unsupported language {value!r}. Use one of: {supported} "
        "(aliases: english, chinese, 英文, 中文)"
    )


def language_label(code: str) -> str:
    return "English" if code == "en" else "中文 (Chinese)"


def response_language_instruction(code: str) -> str:
    """System-prompt suffix forcing model replies in the chosen language."""
    if code == "zh":
        return (
            "Language: Respond entirely in Simplified Chinese (简体中文). "
            "All JSON string values (contributions, objections, reasons, etc.) "
            "must be written in Chinese."
        )
    return (
        "Language: Respond entirely in English. "
        "All JSON string values must be written in English."
    )
