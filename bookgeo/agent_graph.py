"""LangGraph workflow for bookgeo."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from langgraph.graph import StateGraph

from .config import Config, DEFAULT_CONFIG, resolve_output_dir
from .geocode import geocode_candidate
from .ingest import load_text
from .lang_detect import resolve_language
from .models import FictionalPlace, Mention, RealPlace
from .ner import extract_locations
from .pipeline import _write_outputs


class GraphState(dict):
    text: str
    language: str
    mentions: Dict[str, List[Mention]]
    real_places: List[RealPlace]
    fictional_places: List[FictionalPlace]
    output_dir: Path


def load_book_node(state: GraphState) -> GraphState:
    path = state["path"]
    limit_chars = state.get("limit_chars")
    state["text"] = load_text(path, limit_chars=limit_chars)
    return state


def detect_language_node(state: GraphState) -> GraphState:
    lang = state.get("lang")
    state["language"] = resolve_language(state["text"], lang)
    return state


def extract_locations_node(state: GraphState) -> GraphState:
    config: Config = state.get("config", DEFAULT_CONFIG)
    mentions, _ = extract_locations(state["text"], state["language"], config)
    state["mentions"] = mentions
    return state


def geocode_locations_node(state: GraphState) -> GraphState:
    config: Config = state.get("config", DEFAULT_CONFIG)
    mentions = state.get("mentions", {})
    real_places: List[RealPlace] = []
    fictional_places: List[FictionalPlace] = []
    for key, m_list in mentions.items():
        if len(real_places) + len(fictional_places) >= config.max_mentions:
            break
        result, confidence_or_reason = geocode_candidate(key, state["language"], config)
        if not result:
            fictional_places.append(
                FictionalPlace(
                    original_name=m_list[0].text,
                    language=state["language"],
                    mentions=m_list,
                    reason=confidence_or_reason,
                )
            )
            continue
        geometry = result.get("geometry", {}).get("location", {})
        real_places.append(
            RealPlace(
                original_name=m_list[0].text,
                normalized_name=result.get("formatted_address", key),
                latitude=geometry.get("lat"),
                longitude=geometry.get("lng"),
                language=state["language"],
                mentions=m_list,
                confidence=confidence_or_reason,
                raw_geocode=result,
            )
        )
    state["real_places"] = real_places
    state["fictional_places"] = fictional_places
    return state


def save_results_node(state: GraphState) -> GraphState:
    output_dir = resolve_output_dir(state.get("output_dir", "outputs"))
    config: Config = state.get("config", DEFAULT_CONFIG)
    _write_outputs(state.get("real_places", []), state.get("fictional_places", []), output_dir, generate_map=config.generate_map)
    state["output_dir"] = output_dir
    return state


def build_graph(config: Config = DEFAULT_CONFIG):
    graph = StateGraph(GraphState)
    graph.add_node("load_book", load_book_node)
    graph.add_node("detect_language", detect_language_node)
    graph.add_node("extract_locations", extract_locations_node)
    graph.add_node("geocode_locations", geocode_locations_node)
    graph.add_node("save_results", save_results_node)

    graph.set_entry_point("load_book")
    graph.add_edge("load_book", "detect_language")
    graph.add_edge("detect_language", "extract_locations")
    graph.add_edge("extract_locations", "geocode_locations")
    graph.add_edge("geocode_locations", "save_results")
    graph.set_finish_point("save_results")
    return graph.compile()


def run_agent(path: str, output_dir: str, lang: str | None = None, limit_chars: int | None = None, config: Config = DEFAULT_CONFIG):
    workflow = build_graph(config)
    initial_state: GraphState = {
        "path": path,
        "output_dir": output_dir,
        "lang": lang,
        "limit_chars": limit_chars,
        "config": config,
    }
    result_state = workflow.invoke(initial_state)
    return result_state
