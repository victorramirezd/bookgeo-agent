from types import SimpleNamespace

import pytest

from bookgeo.config import Config
from bookgeo.geocode import geocode_candidate, geocode_confidence


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyRequests:
    def __init__(self, payload):
        self.payload = payload
        self.called_with = None

    def get(self, endpoint, params=None, timeout=15):
        self.called_with = SimpleNamespace(endpoint=endpoint, params=params, timeout=timeout)
        return DummyResponse(self.payload)


@pytest.fixture
def config_with_key():
    return Config(google_maps_api_key="test-key")


def test_geocode_candidate_success(monkeypatch, config_with_key):
    payload = {
        "status": "OK",
        "results": [
            {
                "formatted_address": "Paris, France",
                "types": ["locality", "political"],
                "geometry": {"location": {"lat": 48.8566, "lng": 2.3522}},
            }
        ],
    }
    dummy = DummyRequests(payload)
    monkeypatch.setattr("bookgeo.geocode.requests", dummy)
    result, confidence = geocode_candidate("Paris", "en", config_with_key)
    assert result["formatted_address"] == "Paris, France"
    assert confidence == "high"


def test_geocode_candidate_failure(monkeypatch, config_with_key):
    payload = {"status": "ZERO_RESULTS", "results": []}
    dummy = DummyRequests(payload)
    monkeypatch.setattr("bookgeo.geocode.requests", dummy)
    result, reason = geocode_candidate("Imaginary", "en", config_with_key)
    assert result is None
    assert reason == "no geocode result"


def test_geocode_confidence_levels():
    assert geocode_confidence({"types": ["locality"]}) == "high"
    assert geocode_confidence({"types": ["political"]}) == "medium"
    assert geocode_confidence({"types": ["other"]}) == "low"
