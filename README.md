# PlantCare AI — Backend

FastAPI backend for the PlantCare AI Android app. Accepts a base64-encoded plant image and returns structured identification, health assessment, and care guidance powered by GPT-4o vision and a local RAG knowledge base.

---

## **[▶ Watch the App Demo Video](https://drive.google.com/file/d/1EjYOpUo6jgIlb6ii8_AJZaR9JcogyNJj/view?usp=drive_link)**

---

## How It Works

1. The Android app sends a plant image (base64) to `POST /api/v1/analyze`
2. GPT-4o does a quick lightweight identification to form a RAG query
3. The query is embedded and matched against a local vector store (plant profiles, care guides, disease info)
4. GPT-4o vision runs a full structured analysis enriched with the retrieved context
5. A typed `PlantAnalysisResponse` is returned to the app

---

## Project Structure

```
backend/
├── main.py                        # FastAPI app, CORS, routers
├── config.py                      # Settings loaded from .env
├── requirements.txt
├── .env                           # Secret config (not committed)
├── .env.example                   # Template for .env
│
├── api/
│   └── routes/
│       ├── analyze.py             # POST /api/v1/analyze
│       └── health.py              # GET /health
│
├── models/
│   └── plant_response.py          # Pydantic request/response models
│
├── services/
│   ├── llm_service.py             # OpenAI GPT-4o calls (identify + analyze)
│   ├── rag_service.py             # Vector similarity search
│   └── cache_service.py           # In-memory result cache
│
├── rag/
│   ├── setup_knowledge_base.py    # One-time script to build vector store
│   ├── vector_store.json          # Embedded knowledge base (generated)
│   └── knowledge_base/
│       ├── common_plants.json     # Plant profiles
│       ├── diseases.json          # Disease/symptom profiles
│       └── care_guides.txt        # General care guides
│
└── utils/
    ├── error_handlers.py          # Global exception → HTTP response mapping
    └── image_utils.py             # Image resizing and base64 encoding
```

---

## Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o` and `text-embedding-3-small`

---

## Setup & Run

### Step 1 — Clone and enter the backend directory

```bash
cd "Plant Care/backend"
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 3 — Activate the virtual environment

```bash
source .venv/bin/activate
```

> On Windows: `.venv\Scripts\activate`

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-proj-your-key-here
```

### Step 6 — Build the RAG vector store (one-time setup)

This embeds the plant knowledge base using OpenAI embeddings and saves it to `rag/vector_store.json`. Only needs to run once (or again if you update the knowledge base).

```bash
python3 rag/setup_knowledge_base.py
```

Expected output:
```
Embedding plant profiles...
  ✓ 20 plant profiles embedded
Embedding care guides...
  ✓ 40 care guide sections embedded
Embedding disease profiles...
  ✓ 24 disease profiles embedded

✅ Vector store ready: 84 documents saved to rag/vector_store.json
```

### Step 7 — Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server is now accessible at:
- Local: `http://127.0.0.1:8000`
- From Android device on same Wi-Fi: `http://<your-mac-ip>:8000`

To find your Mac's local IP:
```bash
ipconfig getifaddr en0
```

---

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{ "status": "ok", "service": "PlantCare AI" }
```

---

### Analyze Plant

```
POST /api/v1/analyze
Content-Type: application/json
```

Request body:
```json
{
  "image_base64": "<base64-encoded JPEG string>",
  "user_note": "The lower leaves are turning yellow"
}
```

`user_note` is optional.

Response:
```json
{
  "identification": {
    "common_name": "Monstera",
    "scientific_name": "Monstera deliciosa",
    "family": "Araceae",
    "confidence_score": 0.94
  },
  "sunlight": {
    "level": "Bright Indirect Light",
    "hours_per_day": "4-6 hours indirect",
    "tips": "Place 3-5 feet from a south or west-facing window."
  },
  "water": {
    "frequency": "Every 10-14 days",
    "amount": "Water thoroughly until drainage occurs",
    "tips": "Allow top 2 inches to dry before watering again."
  },
  "health": {
    "status": "Healthy",
    "issues_detected": [],
    "improvement_tips": ["Rotate pot every two weeks for even growth"],
    "urgency": "low"
  },
  "fun_facts": ["..."],
  "care_tips": ["..."],
  "rag_sources_used": ["Plant profile: Monstera", "Care guide: Overwatering"]
}
```

Error responses:

| Status | Error code | Meaning |
|--------|-----------|---------|
| 422 | `invalid_image` | Image is not a plant or could not be parsed |
| 429 | `rate_limited` | OpenAI rate limit hit |
| 502 | `llm_unavailable` | OpenAI API error |
| 500 | `internal_error` | Unexpected server error |

---

## Accessing from an Android Device

1. Connect your phone to the **same Wi-Fi network** as your Mac
2. Find your Mac's local IP:
   ```bash
   ipconfig getifaddr en0
   ```
3. Start the server with `--host 0.0.0.0` (already included in the command above)
4. Set the base URL in the Android app to:
   ```
   http://<your-mac-ip>:8000
   ```

---

## Configuration

All settings are in `.env`. Available options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `OPENAI_MODEL` | `gpt-4o` | Model used for plant analysis |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for RAG embeddings |
| `MAX_IMAGE_SIZE_PX` | `1024` | Max image dimension before resizing |
| `CACHE_TTL_SECONDS` | `3600` | How long to cache analysis results |
| `CACHE_MAX_SIZE` | `50` | Max number of cached results |
| `RAG_TOP_K` | `3` | Number of knowledge base chunks to retrieve |
| `RAG_DISTANCE_THRESHOLD` | `0.6` | Cosine similarity cutoff for RAG results |

---

## Interactive API Docs

When the server is running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`