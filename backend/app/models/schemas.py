"""Pydantic v2 schemas shared across the API surface.

These mirror the tables and RPC return shapes in
backend/db/001_init_schema.sql. Where a field exists only in the database
(embeddings, content_tsv) it is deliberately absent here — nothing that
large or internal should cross the API boundary.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

Language = Literal["en", "fr"]
ProvinceCode = Literal["NB", "NS", "PE", "NL"]


# ---------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------

class Province(BaseModel):
    code: ProvinceCode
    name: str


class MunicipalitySource(BaseModel):
    """One source document for a municipality, in one language."""

    language: Language = "en"
    bylaw_name: str
    source_url: str
    source_type: Literal["pdf", "html"] = "pdf"
    document_version: str | None = None
    last_verified_at: date | None = None


class Municipality(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    province_code: ProvinceCode
    name: str
    source_bylaw_name: str
    source_url: str
    languages: list[Language] = Field(default_factory=lambda: ["en"])
    is_active: bool = True

    # Drives the dated disclaimer in Section 5, Rule 6. None means no human
    # has confirmed the source text is current; the UI must render that as
    # "not yet verified" rather than omitting the disclaimer or inventing
    # a date.
    bylaw_last_verified_at: date | None = None


# ---------------------------------------------------------------------
#  Retrieval
# ---------------------------------------------------------------------

class Chunk(BaseModel):
    """A retrieved bylaw chunk, as returned by either search RPC."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    municipality_id: str
    province_code: ProvinceCode
    bylaw_name: str
    section_number: str
    section_title: str | None = None
    chunk_content: str
    language: Language = "en"
    page_number: int | None = None
    metadata: dict = Field(default_factory=dict)

    # Populated by match_bylaw_chunks (similarity) or
    # keyword_search_bylaw_chunks (rank); the merge step in Phase 3 sets
    # `score` to the fused value.
    similarity: float | None = None
    rank: float | None = None
    score: float | None = None

    def citation(self, municipality_name: str) -> str:
        """Render the mandatory inline citation format from Section 5, Rule 3."""
        return f"[{municipality_name} - {self.bylaw_name}, Section {self.section_number}]"


class Citation(BaseModel):
    """Citation surfaced to the frontend for CitationCard.tsx."""

    chunk_id: UUID
    municipality_name: str
    bylaw_name: str
    section_number: str
    section_title: str | None = None
    page_number: int | None = None
    source_url: str
    language: Language = "en"


# ---------------------------------------------------------------------
#  Chat
# ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: Annotated[str, Field(min_length=3, max_length=1000)]
    municipality_id: Annotated[str, Field(min_length=1, max_length=64)]
    province_code: ProvinceCode
    language: Language = "en"

    # Opaque client-generated id used for per-session rate limiting in
    # Phase 5. Not a user identifier and not stored in query_log.
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class StreamEventType(BaseModel):
    """Envelope for a single Server-Sent Event on /stream."""

    type: Literal["token", "citations", "done", "error"]
    data: str | list[Citation] | None = None


# ---------------------------------------------------------------------
#  Health
# ---------------------------------------------------------------------

class DependencyStatus(BaseModel):
    name: str
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    environment: str
    version: str
    checked_at: datetime
    dependencies: list[DependencyStatus] = Field(default_factory=list)
