"""Data models for bookgeo."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Mention(BaseModel):
    text: str
    sentence: str
    start_char: int
    end_char: int
    chunk_id: Optional[int] = None
    label: Optional[str] = None


class RealPlace(BaseModel):
    original_name: str
    normalized_name: str
    latitude: float
    longitude: float
    language: str
    mentions: List[Mention]
    confidence: str = Field(pattern="^(high|medium|low)$")
    raw_geocode: Optional[dict] = None


class FictionalPlace(BaseModel):
    original_name: str
    language: str
    mentions: List[Mention]
    reason: str
