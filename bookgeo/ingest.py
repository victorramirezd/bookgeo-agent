"""Ingestion utilities for bookgeo."""
from __future__ import annotations

from pathlib import Path


class UnsupportedFileError(ValueError):
    """Raised when file type is unsupported."""


def load_text(path: str | Path, limit_chars: int | None = None) -> str:
    """Load UTF-8 text from a .txt file.

    Args:
        path: Path to the file.
        limit_chars: Optional maximum number of characters to return.
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".txt":
        raise UnsupportedFileError("Only .txt files are supported.")
    text = file_path.read_text(encoding="utf-8")
    if limit_chars:
        return text[:limit_chars]
    return text
