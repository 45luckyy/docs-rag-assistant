from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "gemini" uses the hosted API; "offline" uses local hashing embeddings and an
    # extractive answerer so tests/CI run with no key and no network.
    provider: Literal["gemini", "offline"] = "offline"
    google_api_key: str | None = None
    llm_model: str = "gemini-3.5-flash-lite"
    embed_model: str = "gemini-embedding-2-preview"

    docs_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    docs_base_url: str = "https://fastapi.tiangolo.com"

    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 4
    # Refuse when the best chunk's cosine similarity is below this. Tune per
    # provider with `python -m eval.run_eval --sweep`.
    min_score: float = 0.35
    max_search_hops: int = 2

    embed_batch_size: int = 50
    # Free tier counts each chunk as one request (limit 100/min); stay under it.
    embed_per_minute: int = 90


@lru_cache
def get_settings() -> Settings:
    return Settings()
