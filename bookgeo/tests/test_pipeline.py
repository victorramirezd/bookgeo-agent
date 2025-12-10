import pytest

from bookgeo.config import Config
from bookgeo.pipeline import run_pipeline


class DummyConfig(Config):
    def ensure_api_key(self):
        return None


@pytest.fixture
def dummy_config(monkeypatch):
    cfg = DummyConfig(google_maps_api_key="dummy", generate_map=False)
    from bookgeo import geocode as geocode_module

    def fake_extract_locations(text, language, config):
        from bookgeo.models import Mention

        return {"london": [Mention(text="London", sentence=text, start_char=0, end_char=6, chunk_id=0)]}, None

    def fake_geocode_candidate(name, language, config):
        return (
            {
                "formatted_address": f"{name.title()}, Testland",
                "geometry": {"location": {"lat": 1.0, "lng": 2.0}},
                "types": ["locality"],
            },
            "high",
        )

    monkeypatch.setattr(geocode_module, "geocode_candidate", fake_geocode_candidate)
    monkeypatch.setattr("bookgeo.pipeline.geocode_candidate", fake_geocode_candidate)
    monkeypatch.setattr("bookgeo.pipeline.extract_locations", fake_extract_locations)
    return cfg


def test_pipeline_end_to_end(tmp_path, dummy_config):
    sample = tmp_path / "sample.txt"
    sample.write_text("London and Madrid", encoding="utf-8")
    real_places, fictional_places = run_pipeline(str(sample), output_dir=tmp_path, lang="en", config=dummy_config)
    assert len(real_places) >= 1
    assert (tmp_path / "real_places.json").exists()
    assert (tmp_path / "real_places.csv").exists()
