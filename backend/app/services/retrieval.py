"""Hybrid retrieval orchestration (Phase 3, Step 5).

Ties the two retrievers together for the request path: embed the question,
run vector and keyword search concurrently, and fuse the results.

The degradation rule here is the reason this is a module rather than a few
lines in the route. Embedding is a metered third-party call with a hard
free-tier ceiling, and it is the ONLY part of retrieval that can be
rationed. Keyword search runs entirely inside Postgres and costs nothing.

So when embedding fails, the request continues on keyword search alone
rather than returning an error. A keyword-only answer is narrower - it
will miss "granny flat" when the bylaw says "Garden Suite" - but it is
still grounded, still cited, and still correct about what it does find.
Refusing to answer at all, on a public civic tool, because a quota reset
is hours away is a worse outcome than a narrower answer. The degradation
is recorded so it is visible rather than silent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date

import structlog

from app.config import Settings, get_settings
from app.models.schemas import Chunk
from app.services.embedder import Embedder, EmbeddingError
from app.services.keyword_search import KeywordSearch, reciprocal_rank_fusion
from app.services.vector_store import BylawChunkStore

log = structlog.get_logger(__name__)

MUNICIPALITIES_TABLE = "municipalities"


@dataclass
class MunicipalityInfo:
    """Registry facts the request path needs for prompts and citations."""

    id: str
    name: str
    province_code: str
    source_url: str
    bylaw_name: str
    is_active: bool
    languages: list[str] = field(default_factory=lambda: ["en"])
    bylaw_last_verified_at: date | str | None = None


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    vector_hits: int
    keyword_hits: int
    degraded: bool = False
    degraded_reason: str | None = None


class HybridRetriever:
    """Vector + keyword retrieval, fused."""

    def __init__(
        self,
        embedder: Embedder,
        store: BylawChunkStore,
        keyword: KeywordSearch,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._keyword = keyword
        self._settings = settings or get_settings()

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "HybridRetriever":
        settings = settings or get_settings()
        return cls(
            Embedder(settings),
            await BylawChunkStore.create(settings),
            await KeywordSearch.create(settings),
            settings,
        )

    async def municipality(self, municipality_id: str) -> MunicipalityInfo | None:
        """Registry row for one municipality, or None if unknown."""
        response = (
            await self._store._client.table(MUNICIPALITIES_TABLE)
            .select(
                "id,name,province_code,source_url,source_bylaw_name,"
                "is_active,languages,bylaw_last_verified_at"
            )
            .eq("id", municipality_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None

        row = rows[0]
        return MunicipalityInfo(
            id=row["id"],
            name=row["name"],
            province_code=row["province_code"],
            source_url=row.get("source_url") or "",
            bylaw_name=row.get("source_bylaw_name") or "",
            is_active=bool(row.get("is_active")),
            languages=row.get("languages") or ["en"],
            bylaw_last_verified_at=row.get("bylaw_last_verified_at"),
        )

    async def retrieve(
        self,
        query: str,
        municipality_id: str,
        *,
        language: str | None = None,
        limit: int | None = None,
    ) -> RetrievalResult:
        """Embed, search both ways, and fuse."""
        language = language or self._settings.default_language
        limit = limit or (
            self._settings.vector_match_count + self._settings.keyword_match_count
        )

        keyword_task = asyncio.create_task(
            self._keyword.search(query, municipality_id, language=language)
        )

        vector_chunks: list[Chunk] = []
        degraded_reason: str | None = None
        try:
            embedding = await self._embedder.embed_query(query)
            vector_chunks = await self._store.match_chunks(
                embedding, municipality_id, language=language
            )
        except EmbeddingError as exc:
            # See the module docstring: a narrower grounded answer beats no
            # answer. Recorded, not swallowed.
            degraded_reason = f"{type(exc).__name__}: {exc}"
            log.warning(
                "retrieval_degraded",
                municipality=municipality_id,
                detail="embedding unavailable; continuing keyword-only",
                error=degraded_reason,
            )

        keyword_chunks = await keyword_task

        fused = reciprocal_rank_fusion(
            [vector_chunks, keyword_chunks],
            limit=limit,
        )

        log.info(
            "hybrid_retrieval",
            municipality=municipality_id,
            language=language,
            vector=len(vector_chunks),
            keyword=len(keyword_chunks),
            fused=len(fused),
            degraded=degraded_reason is not None,
        )

        return RetrievalResult(
            chunks=fused,
            vector_hits=len(vector_chunks),
            keyword_hits=len(keyword_chunks),
            degraded=degraded_reason is not None,
            degraded_reason=degraded_reason,
        )
