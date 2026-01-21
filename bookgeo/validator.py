"""LangChain-based validation to flag outlier geocodes."""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, List

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import RealPlace


def _extract_country(place: RealPlace) -> str | None:
    comps = place.raw_geocode.get("address_components", []) if place.raw_geocode else []
    for comp in comps:
        if "country" in comp.get("types", []):
            return comp.get("long_name") or comp.get("short_name")
    return None


def _dominant_country(places: Iterable[RealPlace]) -> str | None:
    countries = [_extract_country(p) for p in places]
    countries = [c for c in countries if c]
    if not countries:
        return None
    return Counter(countries).most_common(1)[0][0]


def flag_outliers_langchain(real_places: List[RealPlace], language: str, api_key: str, temperature: float = 0.2) -> List[str]:
    """Use a small LangChain LLM step to flag geocoded places that seem contextually out of place."""
    if not real_places:
        return []
    dominant = _dominant_country(real_places)
    places_summary = [
        {
            "name": p.normalized_name,
            "original": p.original_name,
            "country": _extract_country(p),
            "lat": p.latitude,
            "lng": p.longitude,
            "sentence": p.mentions[0].sentence if p.mentions else "",
        }
        for p in real_places
    ]

    prompt = ChatPromptTemplate.from_template(
        "You are validating geocoded places from one book. The dominant country is likely: {dominant_country}. "
        "Given the list of places, flag the ones that look far away/out-of-context compared to the dominant country "
        "and the sentences. Only flag truly suspicious outliers. "
        "Return a JSON array of place names to review (use the 'name' field). If none, return an empty array.\n\n"
        "Places:\n{places_json}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key)
    chain = prompt | llm
    response = chain.invoke(
        {
            "dominant_country": dominant or "unknown",
            "places_json": json.dumps(places_summary, ensure_ascii=False),
        }
    )
    content = response.content if hasattr(response, "content") else ""
    # Normalize fenced JSON blocks
    fence = re.compile(r"^```(?:json)?\\s*(.*?)\\s*```$", re.DOTALL)
    match = fence.match(content.strip())
    if match:
        content = match.group(1)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        return []
    return []
