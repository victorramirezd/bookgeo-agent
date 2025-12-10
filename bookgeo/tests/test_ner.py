import pytest

from bookgeo.config import DEFAULT_CONFIG
from bookgeo.ner import extract_locations


def _ensure_model(lang: str):
    import spacy

    model = "en_core_web_md" if lang == "en" else "es_core_news_md"
    try:
        spacy.load(model)
    except OSError:
        pytest.skip(f"spaCy model {model} not installed")


def test_extract_locations_en():
    _ensure_model("en")
    text = "London and Paris are major cities."
    mentions, _ = extract_locations(text, "en", DEFAULT_CONFIG)
    assert any("london" == k for k in mentions)
    assert any("paris" == k for k in mentions)


def test_extract_locations_es():
    _ensure_model("es")
    text = "Madrid y Barcelona son ciudades españolas."
    mentions, _ = extract_locations(text, "es", DEFAULT_CONFIG)
    assert any("madrid" == k for k in mentions)
    assert any("barcelona" == k for k in mentions)
