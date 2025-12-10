"""Typer CLI for bookgeo."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .agent_graph import run_agent
from .config import DEFAULT_CONFIG
from .pipeline import run_pipeline
from .utils import save_json

app = typer.Typer(help="Extract and geocode places from books.")


@app.command()
def run(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to UTF-8 .txt book."),
    output_dir: Path = typer.Option("outputs", help="Directory to store outputs."),
    lang: Optional[str] = typer.Option(None, help="Language code (en|es). If omitted, auto-detect."),
    limit_chars: Optional[int] = typer.Option(None, help="Limit characters for quick runs."),
):
    real_places, fictional_places = run_pipeline(
        str(path), output_dir=str(output_dir), lang=lang, limit_chars=limit_chars, config=DEFAULT_CONFIG
    )
    typer.secho(f"Processed {len(real_places)} real places and {len(fictional_places)} fictional entries.", fg=typer.colors.GREEN)


@app.command(name="run-agent")
def run_agent_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to UTF-8 .txt book."),
    output_dir: Path = typer.Option("outputs", help="Directory to store outputs."),
    lang: Optional[str] = typer.Option(None, help="Language code (en|es). If omitted, auto-detect."),
    limit_chars: Optional[int] = typer.Option(None, help="Limit characters for quick runs."),
):
    state = run_agent(str(path), output_dir=str(output_dir), lang=lang, limit_chars=limit_chars, config=DEFAULT_CONFIG)
    real_count = len(state.get("real_places", []))
    fic_count = len(state.get("fictional_places", []))
    typer.secho(f"Agent finished with {real_count} real places and {fic_count} fictional entries.", fg=typer.colors.GREEN)


@app.command()
def inspect(path: Path = typer.Argument(..., exists=True, readable=True, help="Path to real_places.json")):
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(f"Found {len(data)} real places:")
    for entry in data[:10]:
        typer.echo(f"- {entry['original_name']} -> {entry['normalized_name']} ({entry['confidence']})")


if __name__ == "__main__":
    app()
