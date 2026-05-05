from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", override=True)


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    max_image_size_px: int = 1024
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 50
    rag_top_k: int = 3
    rag_distance_threshold: float = 0.6

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
