"""Google Maps geocoding utilities."""
from __future__ import annotations

from typing import Optional
import requests

from .config import Config


def geocode_place(name: str, language: str, config: Config) -> Optional[dict]:
    """Geocode a place name using Google Maps Geocoding API."""
    config.ensure_api_key()
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": name,
        "key": config.google_maps_api_key,
        "language": language,
    }
    response = requests.get(endpoint, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "OK" or not data.get("results"):
        return None
    return data["results"][0]


def geocode_confidence(result: dict) -> str:
    types = result.get("types", [])
    if "locality" in types or "country" in types:
        return "high"
    if "political" in types:
        return "medium"
    return "low"


def geocode_candidate(name: str, language: str, config: Config) -> tuple[Optional[dict], str]:
    result = geocode_place(name, language, config)
    if not result:
        return None, "no geocode result"
    confidence = geocode_confidence(result)
    return result, confidence
