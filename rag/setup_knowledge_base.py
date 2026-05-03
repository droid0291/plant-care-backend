"""
One-time setup script to embed the plant knowledge base into a local JSON vector store.
Run once before starting the server:
    cd backend && python rag/setup_knowledge_base.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
VECTOR_STORE_PATH = Path(__file__).parent / "vector_store.json"


def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def load_plants(client: OpenAI) -> list[dict]:
    plants_file = KNOWLEDGE_BASE_DIR / "common_plants.json"
    with open(plants_file) as f:
        plants = json.load(f)

    docs = []
    for plant in plants:
        text = (
            f"{plant['common_name']} ({plant['scientific_name']}) — Family: {plant['family']}. "
            f"Sunlight: {plant['sunlight']}. Water: {plant['water']}. "
            f"Humidity: {plant['humidity']}. Temperature: {plant['temperature']}. "
            f"Difficulty: {plant['difficulty']}. Toxicity: {plant['toxicity']}. "
            f"Notes: {plant['notes']}"
        )
        docs.append({
            "id": f"plant_{plant['id']}",
            "text": text,
            "embedding": embed_text(client, text),
            "metadata": {
                "type": "plant_profile",
                "common_name": plant["common_name"],
                "scientific_name": plant["scientific_name"],
                "family": plant["family"]
            }
        })
    return docs


def load_care_guides(client: OpenAI) -> list[dict]:
    guides_file = KNOWLEDGE_BASE_DIR / "care_guides.txt"
    with open(guides_file) as f:
        content = f.read()

    sections = [s.strip() for s in content.split("\n\n") if s.strip()]
    docs = []
    for i, section in enumerate(sections):
        title = section.split("\n")[0][:80]
        docs.append({
            "id": f"guide_{i}",
            "text": section,
            "embedding": embed_text(client, section),
            "metadata": {"type": "care_guide", "title": title}
        })
    return docs


def load_diseases(client: OpenAI) -> list[dict]:
    diseases_file = KNOWLEDGE_BASE_DIR / "diseases.json"
    with open(diseases_file) as f:
        diseases = json.load(f)

    docs = []
    for disease in diseases:
        text = (
            f"Symptom: {disease['symptom']}. "
            f"Cause: {disease['likely_cause']}. "
            f"Treatment: {disease['treatment']} "
            f"Urgency: {disease['urgency']}."
        )
        docs.append({
            "id": f"disease_{disease['id']}",
            "text": text,
            "embedding": embed_text(client, text),
            "metadata": {
                "type": "disease",
                "urgency": disease["urgency"],
                "cause": disease["likely_cause"]
            }
        })
    return docs


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    all_docs = []

    print("Embedding plant profiles...")
    plants = load_plants(client)
    all_docs.extend(plants)
    print(f"  ✓ {len(plants)} plant profiles embedded")

    print("Embedding care guides...")
    guides = load_care_guides(client)
    all_docs.extend(guides)
    print(f"  ✓ {len(guides)} care guide sections embedded")

    print("Embedding disease profiles...")
    diseases = load_diseases(client)
    all_docs.extend(diseases)
    print(f"  ✓ {len(diseases)} disease profiles embedded")

    with open(VECTOR_STORE_PATH, "w") as f:
        json.dump(all_docs, f)

    print(f"\n✅ Vector store ready: {len(all_docs)} documents saved to {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    main()
