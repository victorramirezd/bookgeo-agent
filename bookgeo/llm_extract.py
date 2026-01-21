"""LLM-based location extraction."""
from __future__ import annotations

import json
import re
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


# ----------------------------
# Chunking (overlap + snapping)
# ----------------------------

def _snap_end(text: str, start: int, end: int, max_extra: int = 300) -> int:
    """
    Try to extend end to a natural boundary (paragraph/sentence) within max_extra chars.
    This reduces cutting sentences or place mentions mid-way.
    """
    hard_end = min(len(text), end + max_extra)
    window = text[end:hard_end]

    # Prefer paragraph boundary
    m = re.search(r"\n\s*\n", window)
    if m:
        return end + m.end()

    # Otherwise, snap to next sentence end
    m = re.search(r"[.!?]\s", window)
    if m:
        return end + m.end()

    return min(len(text), end)


def _chunk_text(
    text: str,
    chunk_chars: int,
    overlap_chars: int = 350,
) -> List[Tuple[int, int, str]]:
    """
    Overlapping rolling windows:
      - start moves by (chunk_chars - overlap_chars)
      - end snaps forward to sentence/paragraph boundary when possible
    """
    chunks: List[Tuple[int, int, str]] = []
    if not text:
        return chunks

    step = max(1, chunk_chars - max(0, overlap_chars))
    idx = 0
    start = 0

    while start < len(text):
        raw_end = min(len(text), start + chunk_chars)
        end = _snap_end(text, start, raw_end, max_extra=300)
        chunk = text[start:end]
        chunks.append((idx, start, chunk))

        idx += 1
        if end >= len(text):
            break
        start = start + step

    return chunks


# ----------------------------
# Prompting
# ----------------------------

def _build_prompt(language: str, max_items: int) -> str:
    if language == "es":
        return (
            "Tarea: extraer TODAS las ubicaciones y direcciones reales mencionadas en el texto.\n"
            "Incluye (si aparecen): ciudades, países, regiones, ríos, cerros/montañas, monumentos, "
            "parques, iglesias, museos, conventos, puentes, barrios/vecindarios, plazas, calles/avenidas, "
            "y direcciones completas con número (ej: \"Avenida Javier Prado Este 1050, Lima, Perú\").\n\n"
            "Reglas IMPORTANTES:\n"
            "1) No omitas ninguna ubicación mencionada.\n"
            "2) \"name\" debe ser el tramo EXACTO del texto, copiando la mención más larga y específica posible "
            "(conserva números, artículos y nombres de calles). No acortes a solo la ciudad si hay una dirección "
            "o nombre más completo.\n"
            "3) \"sentence\" debe ser la oración del texto que contiene la mención (texto original).\n"
            "4) Contexto de ciudad: si el texto menciona una ciudad (ej: Lima), asume que las menciones cercanas "
            "(plazas, conventos, puentes, parques, etc.) pertenecen a esa ciudad salvo que el texto indique "
            "explícitamente otra ciudad/país.\n"
            "5) Solo lugares reales. No incluyas personas, objetos, organizaciones ni eventos.\n"
            "6) Respuesta: devuelve ÚNICAMENTE un arreglo JSON (sin explicación, sin markdown).\n\n"
            "Formato EXACTO:\n"
            "[{\"name\": \"...\", \"sentence\": \"...\"}, ...]\n\n"
            "Ejemplo:\n"
            "["
            "{\"name\":\"barrio del Rímac\",\"sentence\":\"El barrio del Rímac es histórico.\"},"
            "{\"name\":\"Alameda de los Descalzos\",\"sentence\":\"Caminamos por la Alameda de los Descalzos.\"},"
            "{\"name\":\"Puente de Piedra\",\"sentence\":\"Cruzamos el Puente de Piedra.\"}"
            "]\n\n"
            f"Límite: máximo {max_items} elementos. Si hay más, prioriza direcciones completas y lugares más específicos.\n"
            "Texto a analizar (entre delimitadores):\n"
            "<<<\n"
            "{TEXT}\n"
            ">>>"
        )

    return (
        "Task: extract ALL real geographic locations and addresses mentioned in the text.\n"
        "Include (if present): cities, countries, regions, rivers, mountains, landmarks, parks, churches, museums, "
        "convents, bridges, neighborhoods/barrios, plazas, streets/avenues, and full street addresses with numbers.\n\n"
        "IMPORTANT rules:\n"
        "1) Do not omit any location mentioned.\n"
        "2) \"name\" must be the EXACT span copied from the text, using the longest, most specific span available "
        "(keep street names and numbers). Do not shorten to just the city if a longer address/name exists.\n"
        "3) \"sentence\" must be the original sentence containing the mention.\n"
        "4) City context: if the text mentions a city (e.g., Lima), assume nearby mentions (plazas, bridges, parks, etc.) "
        "belong to that city unless the text clearly switches cities/countries.\n"
        "5) Only real places. No people, objects, organizations, or events.\n"
        "6) Output: return ONLY a JSON array (no explanation, no markdown).\n\n"
        "Exact format:\n"
        "[{\"name\": \"...\", \"sentence\": \"...\"}, ...]\n\n"
        f"Limit: up to {max_items} items. If more exist, prioritize full addresses and more specific places.\n"
        "Text to analyze (between delimiters):\n"
        "<<<\n"
        "{TEXT}\n"
        ">>>"
    )


# ----------------------------
# Parsing helpers
# ----------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", re.IGNORECASE)

def _parse_response(content: str) -> List[dict]:
    """Parse JSON list; robust to fenced blocks; if it fails, return empty."""
    if not content:
        return []

    content = content.strip()

    # If the model wrapped JSON in ```json ... ```
    m = _JSON_FENCE_RE.search(content)
    if m:
        content = m.group(1).strip()

    # Some models prepend/append text; attempt to extract the first JSON array.
    if not content.startswith("["):
        first = content.find("[")
        last = content.rfind("]")
        if first != -1 and last != -1 and last > first:
            content = content[first : last + 1].strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            # Ensure dict elements
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        return []
    return []


def _best_effort_find_span(haystack: str, needle: str) -> int:
    """
    Try to find needle in haystack, preferring exact match, then case-insensitive,
    then a relaxed whitespace match.
    """
    if not needle:
        return -1

    # Exact
    pos = haystack.find(needle)
    if pos != -1:
        return pos

    # Case-insensitive
    lower_h = haystack.lower()
    lower_n = needle.lower()
    pos = lower_h.find(lower_n)
    if pos != -1:
        return pos

    # Relax whitespace (collapse runs)
    def _collapse(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    collapsed_h = _collapse(haystack)
    collapsed_n = _collapse(needle)
    pos = collapsed_h.lower().find(collapsed_n.lower())
    if pos == -1:
        return -1

    # Map back approximately: fallback to first token search in original haystack
    first_token = collapsed_n.split(" ")[0]
    return lower_h.find(first_token.lower())


# ----------------------------
# LLM call
# ----------------------------

def _run_llm_chunk(
    client: OpenAI,
    chunk: str,
    language: str,
    max_items: int,
    temperature: float,
) -> List[dict]:
    prompt = _build_prompt(language, max_items).replace("{TEXT}", chunk)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        # Give room for many items; JSON arrays can be longer than 300 tokens quickly.
        max_tokens=900,
    )

    content = completion.choices[0].message.content or "[]"
    return _parse_response(content)


# ----------------------------
# Public API
# ----------------------------

def extract_locations_llm(
    text: str,
    lang: str | None = None,
    config: Config = DEFAULT_CONFIG,
    chunk_chars: int = 5000,
    temperature: float = 0.1,
) -> LLMExtractResult:
    """Use an LLM to extract location mentions from text."""
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM extraction.")

    language = resolve_language(text, lang)
    client = OpenAI(api_key=config.openai_api_key)

    mentions: Dict[str, List[Mention]] = defaultdict(list)

    # Set max items per chunk high enough to avoid truncation-based false negatives.
    # (Still bounded to avoid runaway output.)
    max_items_per_chunk = max(20, min(80, config.max_mentions))

    # Overlap helps carry city context across chunk boundaries.
    for chunk_idx, offset, chunk in _chunk_text(text, chunk_chars, overlap_chars=350):
        items = _run_llm_chunk(client, chunk, language, max_items_per_chunk, temperature=temperature)

        for item in items:
            name = (item.get("name") or "").strip()
            sentence = (item.get("sentence") or "").strip()
            if not name:
                continue

            key = name.lower()

            # Find best-effort position using name first (more stable than sentence).
            pos = _best_effort_find_span(chunk, name)
            if pos == -1 and sentence:
                pos = _best_effort_find_span(chunk, sentence)

            start_char = offset + (pos if pos != -1 else 0)
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