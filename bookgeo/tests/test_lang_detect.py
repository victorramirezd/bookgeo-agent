from bookgeo.lang_detect import detect_language, resolve_language, UnsupportedLanguageError


def test_detect_language_en():
    text = "This is an English sentence about London."
    assert detect_language(text) == "en"


def test_resolve_language_invalid():
    text = "Ceci est du texte en français."
    try:
        detect_language(text)
    except UnsupportedLanguageError:
        assert True
    else:
        assert False


def test_resolve_language_override():
    text = "Hola desde Madrid"
    assert resolve_language(text, "es") == "es"
    try:
        resolve_language(text, "fr")
    except UnsupportedLanguageError:
        assert True
    else:
        assert False
