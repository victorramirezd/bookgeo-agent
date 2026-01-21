"""Pipeline variant that uses LLM extraction."""
from __future__ import annotations

from collections import Counter
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
from .validator import flag_outliers_langchain


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


def _extract_country_from_geocode(place: RealPlace) -> str | None:
    comps = place.raw_geocode.get("address_components", []) if place.raw_geocode else []
    for comp in comps:
        if "country" in comp.get("types", []):
            return comp.get("long_name") or comp.get("short_name")
    return None


def _dominant_country(real_places: List[RealPlace]) -> str | None:
    countries = [_extract_country_from_geocode(p) for p in real_places]
    countries = [c for c in countries if c]
    if not countries:
        return None
    return Counter(countries).most_common(1)[0][0]


def _reconcile_countries(real_places: List[RealPlace], fictional_places: List[FictionalPlace], language: str, config: Config, dominant: str | None) -> Tuple[List[RealPlace], List[FictionalPlace], List[str]]:
    """Flag places outside dominant country; try re-geocode in dominant country; else move to fictional."""
    if not dominant:
        return real_places, fictional_places, []
    reconciled: List[RealPlace] = []
    updated_fictional = list(fictional_places)
    hard_outliers: List[str] = []

    for p in real_places:
        country = _extract_country_from_geocode(p)
        if country and country != dominant:
            hard_outliers.append(p.normalized_name)
            query = f"{p.original_name} {dominant}"
            retry_result, retry_conf = geocode_candidate(query, language, config)
            if retry_result:
                geometry = retry_result.get("geometry", {}).get("location", {})
                reconciled.append(
                    RealPlace(
                        original_name=p.original_name,
                        normalized_name=retry_result.get("formatted_address", p.normalized_name),
                        latitude=geometry.get("lat"),
                        longitude=geometry.get("lng"),
                        language=language,
                        mentions=p.mentions,
                        confidence=retry_conf,
                        raw_geocode=retry_result,
                    )
                )
            else:
                updated_fictional.append(
                    FictionalPlace(
                        original_name=p.original_name,
                        language=language,
                        mentions=p.mentions,
                        reason="outlier_country_mismatch",
                    )
                )
        else:
            reconciled.append(p)
    return reconciled, updated_fictional, hard_outliers


def run_pipeline_llm(
    path: str,
    output_dir: str,
    lang: str | None = None,
    limit_chars: int | None = None,
    config: Config = DEFAULT_CONFIG,
    chunk_chars: int = 3000,
    temperature: float = 0.0,
) -> Tuple[str, List[RealPlace], List[FictionalPlace], List[str]]:
    """Run pipeline using LLM-based extraction. Returns (language, real_places, fictional_places, outlier_flags)."""
    output_path = resolve_output_dir(output_dir)
    text = load_text(path, limit_chars=limit_chars)
    llm_result = extract_locations_llm(text, lang=lang, config=config, chunk_chars=chunk_chars, temperature=temperature)
    real_places, fictional_places = _mentions_to_places(llm_result.mentions, llm_result.language, config)

    dominant = _dominant_country(real_places)
    real_places, fictional_places, hard_outliers = _reconcile_countries(real_places, fictional_places, llm_result.language, config, dominant)

    outliers: List[str] = list(hard_outliers)
    if config.enable_llm_enhancement and config.openai_api_key:
        try:
            llm_outliers = flag_outliers_langchain(real_places, llm_result.language, api_key=config.openai_api_key, temperature=temperature)
            outliers = list({*outliers, *llm_outliers})
        except Exception:
            outliers = []
        # Always write the file so users can inspect even if empty
        save_json(outliers, output_path / "outlier_places.json")
    _write_outputs(real_places, fictional_places, output_path, generate_map=config.generate_map)
    return llm_result.language, real_places, fictional_places, outliers
