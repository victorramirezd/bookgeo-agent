"""Configuration utilities for bookgeo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

load_dotenv()

SUPPORTED_LANGS = {"en", "es"}


@dataclass
class Config:
    """Runtime configuration for the app."""

    google_maps_api_key: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    enable_llm_enhancement: bool = os.getenv("BOOKGEO_ENABLE_LLM", "false").lower() == "true"
    chunk_size: int = int(os.getenv("BOOKGEO_CHUNK_SIZE", 5000))
    max_mentions: int = int(os.getenv("BOOKGEO_MAX_MENTIONS", 500))
    generate_map: bool = os.getenv("BOOKGEO_GENERATE_MAP", "true").lower() == "true"

    def ensure_api_key(self) -> None:
        if not self.google_maps_api_key:
            raise RuntimeError(
                "GOOGLE_MAPS_API_KEY is required for geocoding. Set it in your environment or .env file."
            )


DEFAULT_CONFIG = Config()


def resolve_output_dir(path: str | Path) -> Path:
    """Ensure output directory exists and return Path."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
