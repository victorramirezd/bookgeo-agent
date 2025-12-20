"""Pipeline variant that uses LLM extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import folium
import pandas as pd

from .config import Config, DEFAULT_CONFIG, resolve_output_dir
from .geocode import geocode_candidate
from .ingest import load_text
from .llm_extract import extract_locations_llm
from .models import FictionalPlace, Mention, RealPlace
from .utils import save_json


def _mentions_to_places(mentions: dict[str, List[Mention]], language: str, config: Config) -> Tuple[List[RealPlace], List[FictionalPlace]]:
    real_places: List[RealPlace] = []
    fictional_places: List[FictionalPlace] = []

    for key, m_list in mentions.items():
        if len(real_places) + len(fictional_places) >= config.max_mentions:
            break
        result, confidence_or_reason = geocode_candidate(key, language, config)
        if not result:
            fictional_places.append(
                FictionalPlace(
                    original_name=m_list[0].text,
                    language=language,
                    mentions=m_list,
                    reason=confidence_or_reason,
                )
            )
            continue
        geometry = result.get("geometry", {}).get("location", {})
        real_places.append(
            RealPlace(
                original_name=m_list[0].text,
                normalized_name=result.get("formatted_address", key),
                latitude=geometry.get("lat"),
                longitude=geometry.get("lng"),
                language=language,
                mentions=m_list,
                confidence=confidence_or_reason,
                raw_geocode=result,
            )
        )
    return real_places, fictional_places


def _write_outputs(real_places: Iterable[RealPlace], fictional_places: Iterable[FictionalPlace], output_dir: Path, generate_map: bool = True) -> None:
    real_list = list(real_places)
    fic_list = list(fictional_places)
    save_json([p.model_dump() for p in real_list], output_dir / "real_places.json")
    save_json([p.model_dump() for p in fic_list], output_dir / "fictional_places.json")

    df = pd.DataFrame([
        {
            "original_name": p.original_name,
            "normalized_name": p.normalized_name,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "language": p.language,
            "confidence": p.confidence,
        }
        for p in real_list
    ])
    df.to_csv(output_dir / "real_places.csv", index=False)

    if generate_map and real_list:
        m = folium.Map(location=[real_list[0].latitude, real_list[0].longitude], zoom_start=2)
        for p in real_list:
            folium.Marker(
                location=[p.latitude, p.longitude],
                popup=f"{p.normalized_name} ({p.confidence})",
            ).add_to(m)
        m.save(output_dir / "places_map.html")


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
