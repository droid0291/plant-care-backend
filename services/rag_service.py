import json
from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np
from openai import OpenAI
from config import get_settings
from models.plant_response import (
    PlantAnalysisResponse, PlantIdentification,
    SunlightRequirement, WaterRequirement, PlantHealthOnly,
)

VECTOR_STORE_PATH = Path(__file__).parent.parent / "rag" / "vector_store.json"
PLANTS_FILE = Path(__file__).parent.parent / "rag" / "knowledge_base" / "common_plants.json"


@lru_cache(maxsize=1)
def _load_vector_store() -> tuple[list[dict], np.ndarray]:
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}. "
            "Run: python rag/setup_knowledge_base.py"
        )
    with open(VECTOR_STORE_PATH) as f:
        docs = json.load(f)
    embeddings = np.array([doc["embedding"] for doc in docs], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return docs, embeddings


@lru_cache(maxsize=1)
def _load_plant_profiles() -> list[dict]:
    with open(PLANTS_FILE) as f:
        return json.load(f)


def _embed_query(query: str, openai_client: OpenAI) -> np.ndarray:
    settings = get_settings()
    response = openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=query
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
    return vec


def query_knowledge_base(plant_query: str, openai_client: OpenAI) -> tuple[list[str], list[str]]:
    """Return (chunks, labels) for the top-k relevant KB entries — single embedding call."""
    settings = get_settings()
    docs, embeddings = _load_vector_store()
    query_vec = _embed_query(plant_query, openai_client)

    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:settings.rag_top_k]
    min_similarity = 1.0 - settings.rag_distance_threshold

    chunks, labels = [], []
    for i in top_indices:
        if scores[i] < min_similarity:
            continue
        chunks.append(docs[i]["text"])
        meta = docs[i]["metadata"]
        doc_type = meta.get("type", "unknown")
        if doc_type == "plant_profile":
            labels.append(f"Plant profile: {meta.get('common_name', 'Unknown')}")
        elif doc_type == "care_guide":
            labels.append(f"Care guide: {meta.get('title', 'Unknown')[:50]}")
        elif doc_type == "disease":
            labels.append(f"Disease info: {meta.get('cause', 'Unknown')[:50]}")

    return chunks, labels


def lookup_plant_by_name(name: str) -> Optional[dict]:
    """Direct lookup in common_plants.json by common or scientific name. No API call needed."""
    plants = _load_plant_profiles()
    normalized = name.lower().strip()
    for plant in plants:
        common = plant["common_name"].lower()
        scientific = plant["scientific_name"].lower()
        if normalized == common or normalized == scientific:
            return plant
        if normalized in common or common in normalized:
            return plant
    return None


def _derive_sunlight_hours(sunlight_str: str) -> str:
    s = sunlight_str.lower()
    if "full sun" in s:
        return "6-8 hours direct"
    if "bright indirect" in s:
        return "4-6 hours indirect"
    if "low" in s or "shade" in s:
        return "1-3 hours indirect"
    if "indirect" in s:
        return "2-6 hours indirect"
    return "Varies — see care profile"


def build_response_from_kb(kb_plant: dict, health_result: PlantHealthOnly) -> PlantAnalysisResponse:
    """Assemble a full PlantAnalysisResponse from KB data + GPT-4o health assessment."""
    water_parts = kb_plant["water"].split(";", 1)
    water_frequency = water_parts[0].strip()
    water_tip = water_parts[1].strip() if len(water_parts) > 1 else "Check soil moisture before watering."

    return PlantAnalysisResponse(
        identification=PlantIdentification(
            common_name=kb_plant["common_name"],
            scientific_name=kb_plant["scientific_name"],
            family=kb_plant["family"],
            confidence_score=health_result.confidence_score,
        ),
        sunlight=SunlightRequirement(
            level=kb_plant["sunlight"],
            hours_per_day=_derive_sunlight_hours(kb_plant["sunlight"]),
            tips=f"Avoid direct sun unless the profile specifies full sun. {kb_plant['sunlight'].split(';')[0]}.",
        ),
        water=WaterRequirement(
            frequency=water_frequency,
            amount="Water thoroughly until it drains from the pot",
            tips=water_tip,
        ),
        health=health_result.health,
        fun_facts=health_result.fun_facts,
        care_tips=[kb_plant["notes"]],
        rag_sources_used=[f"Plant profile: {kb_plant['common_name']}"],
    )