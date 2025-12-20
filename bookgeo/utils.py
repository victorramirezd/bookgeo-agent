"""Misc helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import FictionalPlace, RealPlace


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_places(real_places: Iterable[RealPlace], fictional_places: Iterable[FictionalPlace], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    real_path = output_dir / "real_places.json"
    fictional_path = output_dir / "fictional_places.json"
    save_json([place.model_dump() for place in real_places], real_path)
    save_json([place.model_dump() for place in fictional_places], fictional_path)
    return {"real": real_path, "fictional": fictional_path}
