"""LLM-based location extraction."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from openai import OpenAI

from .config import Config, DEFAULT_CONFIG
from .lang_detect import resolve_language
from .models import Mention


@dataclass
class LLMExtractResult:
    language: str
    mentions: Dict[str, List[Mention]]


def _chunk_text(text: str, chunk_chars: int) -> List[Tuple[int, int, str]]:
    chunks: List[Tuple[int, int, str]] = []
    for idx, i in enumerate(range(0, len(text), chunk_chars)):
        chunks.append((idx, i, text[i : i + chunk_chars]))
    return chunks


def _parse_response(content: str) -> List[dict]:
    """Parse JSON list; if it fails, return empty."""
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _run_llm_chunk(client: OpenAI, chunk: str, language: str, max_items: int) -> List[dict]:
    prompt = (
        "Extract real geographic locations and addresses mentioned in the text. Include cities, countries, regions, "
        "rivers, landmarks, streets, and full street addresses with numbers when present "
        "(e.g., 'Avenida Javier Prado este 1050, Lima, Peru'). "
        "Return a JSON array. Each item must include: "
        "{\"name\": exact span copied from the text (use the longest specific span available; keep numbers/street names), "
        "\"sentence\": the sentence containing the mention}. "
        "Do not shorten to just the city if a longer address span is present. "
        f"Language: {language}. Limit to {max_items} items. Only real places, no people or objects."
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        {"role": "user", "content": chunk},
    ],
    temperature=0.0,
    max_tokens=300,
)
    content = completion.choices[0].message.content or "[]"
    return _parse_response(content)


def extract_locations_llm(
    text: str,
    lang: str | None = None,
    config: Config = DEFAULT_CONFIG,
    chunk_chars: int = 3000,
) -> LLMExtractResult:
    """Use an LLM to extract location mentions from text."""
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM extraction.")

    language = resolve_language(text, lang)
    client = OpenAI(api_key=config.openai_api_key)

    mentions: Dict[str, List[Mention]] = defaultdict(list)
    max_items_per_chunk = max(5, min(30, config.max_mentions))
    for chunk_idx, offset, chunk in _chunk_text(text, chunk_chars):
        items = _run_llm_chunk(client, chunk, language, max_items_per_chunk)
        for item in items:
            name = (item.get("name") or "").strip()
            sentence = (item.get("sentence") or "").strip()
            if not name:
                continue
            key = name.lower()
            # best-effort position lookup (case-insensitive)
            lower_chunk = chunk.lower()
            pos = lower_chunk.find(sentence.lower()) if sentence else lower_chunk.find(name.lower())
            start_char = offset + pos if pos != -1 else offset
            end_char = start_char + len(name)
            mentions[key].append(
                Mention(
                    text=name,
                    sentence=sentence or name,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_id=chunk_idx,
                    label="LLM",
                )
            )
            if sum(len(v) for v in mentions.values()) >= config.max_mentions:
                break
        if sum(len(v) for v in mentions.values()) >= config.max_mentions:
            break
    return LLMExtractResult(language=language, mentions=mentions)
