"""Application settings, loaded from the environment via Pydantic Settings.

Phase 0 locked the deployment target as local-only, so defaults here point
at localhost. Supabase is managed regardless of where the app runs, so its
credentials are required even in local development.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Language = Literal["en", "fr"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Runtime -----------------------------------------------------
    environment: Literal["local", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ---- Supabase ----------------------------------------------------
    # The backend uses the service-role key, which bypasses Row Level
    # Security. It must never be exposed to the browser; the frontend gets
    # the anon key, which can read only provinces and active municipalities
    # (see PART 7 of backend/db/001_init_schema.sql).
    supabase_url: str
    supabase_service_role_key: SecretStr
    supabase_anon_key: SecretStr | None = None

    # ---- Model providers ---------------------------------------------
    # Embeddings run on Gemini rather than the OpenAI model named in
    # Section 1, at user direction: the free tier needs no prepayment.
    # 1536 is a supported Matryoshka output size, so VECTOR(1536) and the
    # cosine index are unaffected by the switch.
    gemini_api_key: SecretStr | None = None
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 1536

    # Gemini's free tier counts each embedded TEXT against a per-minute
    # quota (observed 100/min), not each HTTP request, so the ingestion
    # pipeline paces itself. Raise this on a paid tier.
    embedding_items_per_minute: int = 100

    # Retained so an OpenAI key can be reinstated without a code change.
    openai_api_key: SecretStr | None = None

    groq_api_key: SecretStr | None = None
    # Section 1 names llama-3.3-70b-versatile, which Groq has since
    # removed entirely (the API returns model_not_found). Replaced at
    # user direction after checking the live model list.
    llm_model: str = "qwen/qwen3.8-27b"
    llm_temperature: float = 0.1
    # Groq's free tier caps qwen at 1000 OUTPUT tokens per minute, and
    # rejects a request up front whose max_tokens exceeds that ceiling
    # rather than truncating it. 800 leaves headroom for the reasoning
    # tokens this model also counts. Raise it on a paid tier.
    llm_max_tokens: int = 800

    # ---- Retrieval ---------------------------------------------------
    # Mirrors the RPC defaults in 001_init_schema.sql. Keep them in step:
    # if these drift from the SQL defaults, tuning one silently does nothing.
    match_threshold: float = 0.3
    vector_match_count: int = 5
    keyword_match_count: int = 5

    # ---- Localisation -------------------------------------------------
    default_language: Language = "en"
    supported_languages: list[Language] = ["en", "fr"]

    # ---- Rate limiting (Section 1; enforced in Phase 5) ---------------
    rate_limit_per_minute: int = 10
    rate_limit_per_day: int = 200

    # ---- CORS ---------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("supabase_url")
    @classmethod
    def _validate_supabase_url(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("SUPABASE_URL must start with https://")
        return v.rstrip("/")

    @field_validator("embedding_dimensions")
    @classmethod
    def _validate_dimensions(cls, v: int) -> int:
        # bylaw_chunks.embedding is declared VECTOR(1536). A mismatch here
        # fails at insert time with an opaque pgvector error, so catch it
        # at startup instead.
        if v != 1536:
            raise ValueError(
                "embedding_dimensions must be 1536 to match VECTOR(1536) "
                "in bylaw_chunks.embedding"
            )
        return v

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()  # type: ignore[call-arg]
