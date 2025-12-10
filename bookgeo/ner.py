"""Location extraction using spaCy."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .config import Config
from .models import Mention

LOCATION_LABELS = {"GPE", "LOC", "FAC", "ORG"}


_MODEL_NAMES = {
    "en": "en_core_web_md",
    "es": "es_core_news_md",
}


def _load_model(lang: str) -> Language:
    model_name = _MODEL_NAMES[lang]
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Install with: python -m spacy download {model_name}"
        ) from exc


def _iter_chunks(text: str, chunk_size: int) -> List[Tuple[int, int, str]]:
    chunks: List[Tuple[int, int, str]] = []
    for idx, i in enumerate(range(0, len(text), chunk_size)):
        chunks.append((idx, i, text[i : i + chunk_size]))
    return chunks


def extract_locations(text: str, language: str, config: Config) -> Tuple[Dict[str, List[Mention]], Doc]:
    """Extract location mentions from text."""
    nlp = _load_model(language)
    chunks = _iter_chunks(text, config.chunk_size)
    mentions: Dict[str, List[Mention]] = defaultdict(list)
    doc_parts: List[Doc] = []

    for chunk_index, start_offset, chunk in chunks:
        doc = nlp(chunk)
        doc_parts.append(doc)
        for ent in doc.ents:
            if ent.label_ not in LOCATION_LABELS:
                continue
            key = ent.text.lower()
            sentence = ent.sent.text if ent.sent else ent.text
            mentions[key].append(
                Mention(
                    text=ent.text,
                    sentence=sentence,
                    start_char=start_offset + ent.start_char,
                    end_char=start_offset + ent.end_char,
                    chunk_id=chunk_index,
                    label=ent.label_,
                )
            )
    combined_doc = Doc.from_docs([d for d in doc_parts]) if doc_parts else nlp("")
    return mentions, combined_doc
