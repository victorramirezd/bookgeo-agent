"""Pipeline variant that uses LLM extraction instead of spaCy."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .config import Config, DEFAULT_CONFIG, resolve_output_dir
from .ingest import load_text
from .llm_extract import extract_locations_llm
from .models import FictionalPlace, RealPlace
from .pipeline import _mentions_to_places, _write_outputs


def run_pipeline_llm(
    path: str,
    output_dir: str,
    lang: str | None = None,
    limit_chars: int | None = None,
    config: Config = DEFAULT_CONFIG,
    chunk_chars: int = 3000,
) -> Tuple[List[RealPlace], List[FictionalPlace]]:
    """Run pipeline using LLM-based extraction."""
    output_path = resolve_output_dir(output_dir)
    text = load_text(path, limit_chars=limit_chars)
    llm_result = extract_locations_llm(text, lang=lang, config=config, chunk_chars=chunk_chars)
    real_places, fictional_places = _mentions_to_places(llm_result.mentions, llm_result.language, config)
    _write_outputs(real_places, fictional_places, output_path, generate_map=config.generate_map)
    return real_places, fictional_places
