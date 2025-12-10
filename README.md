# bookgeo

Local AI agent that extracts and geocodes place references from English or Spanish books using spaCy NER and Google Maps Geocoding. Includes a Typer CLI, optional LangGraph workflow, and utilities for exporting JSON/CSV/HTML outputs.

## Features
- Supports **English (en)** and **Spanish (es)** only. Rejects other languages.
- Language auto-detection via `langdetect` when `--lang` is omitted.
- spaCy NER (en_core_web_md, es_core_news_md) with chunked processing for long texts.
- Geocoding via Google Maps Geocoding API with a simple confidence heuristic.
- Outputs real vs fictional/unresolved places (JSON + CSV, optional folium map).
- LangGraph orchestration option wrapping the pure Python pipeline.
- CLI built with Typer and packaged entrypoint `bookgeo`.

## macOS (Apple Silicon) setup with VS Code
1. Install Python 3.9+ (e.g., via `pyenv`).
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install Poetry and project dependencies:
   ```bash
   pip install poetry
   poetry install
   ```
4. Install spaCy language models (CPU only):
   ```bash
   python -m spacy download en_core_web_md
   python -m spacy download es_core_news_md
   ```
5. Copy `.env.example` to `.env` and fill `GOOGLE_MAPS_API_KEY` (and optional `OPENAI_API_KEY`).
6. Open the folder in VS Code and select the `.venv` interpreter. Recommended extensions: Python, Pylance.

### VS Code debugging (CLI)
An example `.vscode/launch.json` is included. After activating the venv in VS Code, you can run the `bookgeo` CLI with the debugger using the **Run and Debug** panel.

## CLI usage
```bash
bookgeo run path/to/book.txt --output-dir outputs/
bookgeo run-agent path/to/book.txt --output-dir outputs/  # LangGraph workflow
bookgeo inspect outputs/real_places.json
```
Options:
- `--lang` : `en` or `es`. If omitted, the language is detected; non-en/es abort.
- `--limit-chars` : process only a prefix of the text (useful for tests).

## Outputs
- `real_places.json`: list of resolved places with normalized name/address, lat/lng, language, confidence, and mentions.
- `fictional_places.json`: unresolved candidates with reason.
- `real_places.csv`: flat table.
- `places_map.html`: (optional) folium map with markers.

## Architecture
```
bookgeo/
  config.py        # env + defaults
  ingest.py        # text loading
  lang_detect.py   # language detection and validation
  ner.py           # spaCy NER location extraction
  geocode.py       # Google Maps geocoding helpers
  pipeline.py      # synchronous pipeline
  agent_graph.py   # LangGraph workflow wrapper
  cli.py           # Typer CLI entrypoint
  models.py        # Pydantic models
```
LangGraph nodes: load_book -> detect_language -> extract_locations -> geocode_locations -> save_results.

## Testing
Run pytest (spaCy model-dependent tests are skipped if models are missing):
```bash
poetry run pytest
```

## Samples
- `bookgeo/samples/sample_book_en.txt`
- `bookgeo/samples/sample_book_es.txt`

## Environment variables
See `.env.example`:
- `GOOGLE_MAPS_API_KEY` (required for geocoding)
- `OPENAI_API_KEY` (optional, for future LLM steps)
- `BOOKGEO_ENABLE_LLM` (optional bool, default false)
