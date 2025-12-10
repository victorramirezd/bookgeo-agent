"""Language detection utilities."""
from __future__ import annotations

from langdetect import detect

from .config import SUPPORTED_LANGS


class UnsupportedLanguageError(ValueError):
    """Raised when detected or provided language is not supported."""


def detect_language(text: str) -> str:
    """Detect language code using langdetect.

    Returns "en" or "es". Raises UnsupportedLanguageError otherwise.
    """
    lang = detect(text) if text.strip() else ""
    if lang not in SUPPORTED_LANGS:
        raise UnsupportedLanguageError(
            f"Detected language '{lang}' is not supported. Only en/es are allowed."
        )
    return lang


def resolve_language(text: str, provided: str | None) -> str:
    """Return valid language code based on provided flag or detection."""
    if provided:
        if provided not in SUPPORTED_LANGS:
            raise UnsupportedLanguageError(
                f"Language '{provided}' is not supported. Use 'en' or 'es'."
            )
        return provided
    return detect_language(text)
