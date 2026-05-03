from __future__ import annotations
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from models.plant_response import PlantAnalysisResponse
from config import get_settings

# In-memory store: hash -> (response, timestamp)
_cache: dict[str, tuple[PlantAnalysisResponse, datetime]] = {}


def _cache_key(image_base64: str) -> str:
    return hashlib.sha256(image_base64.encode()).hexdigest()


def get_cached(image_base64: str) -> Optional[PlantAnalysisResponse]:
    settings = get_settings()
    key = _cache_key(image_base64)
    if key not in _cache:
        return None
    result, timestamp = _cache[key]
    if datetime.now() - timestamp > timedelta(seconds=settings.cache_ttl_seconds):
        del _cache[key]
        return None
    return result


def store_cached(image_base64: str, result: PlantAnalysisResponse) -> None:
    settings = get_settings()
    key = _cache_key(image_base64)
    # Evict oldest entry if at capacity
    if len(_cache) >= settings.cache_max_size:
        oldest_key = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest_key]
    _cache[key] = (result, datetime.now())


def cache_size() -> int:
    return len(_cache)
