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
    # Embeddings run on Voyage rather than the OpenAI model named in
    # Section 1, at user direction. Its 200M-token free grant has no daily
    # cap (Gemini's free tier allowed 1,000/day, and one bilingual
    # municipality is ~1,340 chunks), and at 0.28s per query it is half
    # the latency of the local model that was tried in between.
    voyage_api_key: SecretStr | None = None
    embedding_model: str = "voyage-4-large"
    embedding_dimensions: int = 1024

    # Voyage throttles accounts with no payment method on file to 3 RPM
    # and 10K TPM. The 200M-token grant is still free at that tier - only
    # the rate is limited - so ingestion paces itself rather than failing.
    # Adding a payment method raises these substantially and still spends
    # the free grant first; raise both here if that is done.
    embedding_requests_per_minute: int = 3
    embedding_tokens_per_minute: int = 10_000

    # Retained so either previous provider can be reinstated without a
    # code change; unused while embeddings run on Voyage.
    gemini_api_key: SecretStr | None = None

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
        # bylaw_chunks.embedding is declared VECTOR(1024) as of migration
        # 003. A mismatch fails at insert time with an opaque pgvector
        # error, so catch it at startup instead.
        if v != 1024:
            raise ValueError(
                "embedding_dimensions must be 1024 to match VECTOR(1024) "
                "in bylaw_chunks.embedding (see db/003_local_embeddings.sql)"
            )
        return v

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()  # type: ignore[call-arg]
