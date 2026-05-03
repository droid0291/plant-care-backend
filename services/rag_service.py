import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from openai import OpenAI
from config import get_settings

VECTOR_STORE_PATH = Path(__file__).parent.parent / "rag" / "vector_store.json"


@lru_cache(maxsize=1)
def _load_vector_store() -> tuple[list[dict], np.ndarray]:
    """Load vector store from disk once, cache in memory."""
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}. "
            "Run: python rag/setup_knowledge_base.py"
        )
    with open(VECTOR_STORE_PATH) as f:
        docs = json.load(f)
    embeddings = np.array([doc["embedding"] for doc in docs], dtype=np.float32)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return docs, embeddings


def _embed_query(query: str, openai_client: OpenAI) -> np.ndarray:
    settings = get_settings()
    response = openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=query
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
    return vec


def query_plant_knowledge(plant_query: str, openai_client: OpenAI) -> list[str]:
    """Return top-k relevant knowledge chunks as plain text."""
    settings = get_settings()
    docs, embeddings = _load_vector_store()
    query_vec = _embed_query(plant_query, openai_client)

    # Cosine similarity = dot product of normalized vectors
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:settings.rag_top_k]

    # Convert cosine distance threshold: similarity > (1 - threshold)
    min_similarity = 1.0 - settings.rag_distance_threshold
    return [
        docs[i]["text"]
        for i in top_indices
        if scores[i] >= min_similarity
    ]


def get_rag_source_labels(plant_query: str, openai_client: OpenAI) -> list[str]:
    """Return human-readable labels of which KB sections were retrieved."""
    settings = get_settings()
    docs, embeddings = _load_vector_store()
    query_vec = _embed_query(plant_query, openai_client)

    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:settings.rag_top_k]

    min_similarity = 1.0 - settings.rag_distance_threshold
    labels = []
    for i in top_indices:
        if scores[i] < min_similarity:
            continue
        meta = docs[i]["metadata"]
        doc_type = meta.get("type", "unknown")
        if doc_type == "plant_profile":
            labels.append(f"Plant profile: {meta.get('common_name', 'Unknown')}")
        elif doc_type == "care_guide":
            labels.append(f"Care guide: {meta.get('title', 'Unknown')[:50]}")
        elif doc_type == "disease":
            labels.append(f"Disease info: {meta.get('cause', 'Unknown')[:50]}")
    return labels
