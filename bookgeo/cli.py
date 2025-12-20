"""Typer CLI for bookgeo."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import DEFAULT_CONFIG
from .llm_pipeline import run_pipeline_llm

app = typer.Typer(help="Extract and geocode places from books.")


@app.command()
def inspect(path: Path = typer.Argument(..., exists=True, readable=True, help="Path to real_places.json")):
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(f"Found {len(data)} real places:")
    for entry in data[:10]:
        typer.echo(f"- {entry['original_name']} -> {entry['normalized_name']} ({entry['confidence']})")


@app.command(name="run")
def run(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to UTF-8 .txt book."),
    output_dir: Path = typer.Option("outputs", help="Directory to store outputs."),
    lang: Optional[str] = typer.Option(None, help="Language code (en|es). If omitted, auto-detect."),
    limit_chars: Optional[int] = typer.Option(None, help="Limit characters for quick runs."),
    chunk_chars: Optional[int] = typer.Option(3000, help="Chunk size (chars) for LLM extraction."),
):
    """LLM-based extraction of place mentions (uses OpenAI)."""
    real_places, fictional_places = run_pipeline_llm(
        str(path),
        output_dir=str(output_dir),
        lang=lang,
        limit_chars=limit_chars,
        config=DEFAULT_CONFIG,
        chunk_chars=chunk_chars,
    )
    typer.secho(
        f"LLM run processed {len(real_places)} real places and {len(fictional_places)} fictional entries.",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
