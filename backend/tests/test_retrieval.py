"""Unit tests for app.services.retrieval (Phase 3, Step 5).

Hermetic: embedder, vector store and keyword search are all stubs.

The degradation cases are the point of this file. Embedding is the only
rationed part of retrieval, and its free-tier quota does run out in
practice - it did so repeatedly during Phase 2. What the app does at that
moment is a product decision, not an implementation detail.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.models.schemas import Chunk
from app.services.embedder import EmbeddingQuotaExhausted
from app.services.retrieval import HybridRetriever, MunicipalityInfo


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        supabase_url="https://test-placeholder.supabase.co",
        supabase_service_role_key="test-placeholder",
    )


def chunk(section: str, *, similarity=None, rank=None, id_: UUID | None = None) -> Chunk:
    return Chunk(
        id=id_ or uuid4(),
        municipality_id="nb_fredericton",
        province_code="NB",
        bylaw_name="Zoning By-law Z-5",
        section_number=section,
        chunk_content="body",
        similarity=similarity,
        rank=rank,
    )


class StubEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def embed_query(self, text: str):
        self.calls += 1
        if self.error:
            raise self.error
        return [0.1] * 1024


class StubStore:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or []
        self.calls = 0

    async def match_chunks(self, embedding, municipality_id, **kwargs):
        self.calls += 1
        return self.chunks


class StubKeyword:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks or []
        self.calls = 0

    async def search(self, query, municipality_id, **kwargs):
        self.calls += 1
        return self.chunks


def make_retriever(embedder=None, store=None, keyword=None) -> HybridRetriever:
    return HybridRetriever(
        embedder or StubEmbedder(),  # type: ignore[arg-type]
        store or StubStore(),  # type: ignore[arg-type]
        keyword or StubKeyword(),  # type: ignore[arg-type]
        settings(),
    )


def retrieve(retriever, query="are kennels allowed"):
    return asyncio.run(retriever.retrieve(query, "nb_fredericton", language="en"))


# ---------------------------------------------------------------------
#  Normal operation
# ---------------------------------------------------------------------


def test_both_retrievers_are_consulted():
    store = StubStore([chunk("8.14(2)", similarity=0.74)])
    keyword = StubKeyword([chunk("3(107)", rank=0.8)])
    result = retrieve(make_retriever(store=store, keyword=keyword))

    assert store.calls == 1
    assert keyword.calls == 1
    assert result.vector_hits == 1
    assert result.keyword_hits == 1
    assert result.degraded is False


def test_results_from_both_sides_are_fused():
    store = StubStore([chunk("8.14(2)", similarity=0.74)])
    keyword = StubKeyword([chunk("3(107)", rank=0.8)])
    sections = {c.section_number for c in retrieve(make_retriever(store=store, keyword=keyword)).chunks}
    assert sections == {"8.14(2)", "3(107)"}


def test_a_chunk_found_by_both_appears_once():
    shared = uuid4()
    store = StubStore([chunk("8.14(2)", similarity=0.74, id_=shared)])
    keyword = StubKeyword([chunk("8.14(2)", rank=1.0, id_=shared)])
    assert len(retrieve(make_retriever(store=store, keyword=keyword)).chunks) == 1


# ---------------------------------------------------------------------
#  Degradation
# ---------------------------------------------------------------------


def test_exhausted_embedding_quota_falls_back_to_keyword_only():
    """A narrower grounded answer beats refusing to answer at all."""
    embedder = StubEmbedder(error=EmbeddingQuotaExhausted("daily quota spent"))
    keyword = StubKeyword([chunk("8.14(2)", rank=1.0)])

    result = retrieve(make_retriever(embedder=embedder, keyword=keyword))

    assert result.degraded is True
    assert result.vector_hits == 0
    assert result.keyword_hits == 1
    assert [c.section_number for c in result.chunks] == ["8.14(2)"]


def test_degradation_is_recorded_not_silent():
    embedder = StubEmbedder(error=EmbeddingQuotaExhausted("daily quota spent"))
    result = retrieve(make_retriever(embedder=embedder, keyword=StubKeyword([chunk("8.1(1)", rank=0.5)])))
    assert "EmbeddingQuotaExhausted" in (result.degraded_reason or "")


def test_vector_search_is_skipped_when_embedding_fails():
    """No point calling the RPC with no vector to search on."""
    embedder = StubEmbedder(error=EmbeddingQuotaExhausted("spent"))
    store = StubStore([chunk("8.14(2)", similarity=0.7)])
    retrieve(make_retriever(embedder=embedder, store=store))
    assert store.calls == 0


def test_degraded_retrieval_with_no_keyword_hits_returns_nothing():
    """Which the engine then answers with the Rule 1 fallback."""
    embedder = StubEmbedder(error=EmbeddingQuotaExhausted("spent"))
    result = retrieve(make_retriever(embedder=embedder, keyword=StubKeyword([])))
    assert result.chunks == []
    assert result.degraded is True


def test_an_unexpected_embedding_error_also_degrades():
    from app.services.embedder import EmbeddingError

    embedder = StubEmbedder(error=EmbeddingError("provider unreachable"))
    result = retrieve(make_retriever(embedder=embedder, keyword=StubKeyword([chunk("8.1(1)", rank=0.5)])))
    assert result.degraded is True
    assert len(result.chunks) == 1


def test_a_non_embedding_failure_is_not_swallowed():
    """Only embedding is treated as optional; a store fault is a real fault."""

    class BrokenStore:
        async def match_chunks(self, *args, **kwargs):
            raise RuntimeError("postgrest down")

    with pytest.raises(RuntimeError, match="postgrest down"):
        retrieve(make_retriever(store=BrokenStore()))


# ---------------------------------------------------------------------
#  Municipality info
# ---------------------------------------------------------------------


def test_municipality_info_defaults_are_safe():
    info = MunicipalityInfo(
        id="nb_fredericton",
        name="Fredericton",
        province_code="NB",
        source_url="https://example.ca/z5.pdf",
        bylaw_name="Zoning By-law Z-5",
        is_active=True,
    )
    assert info.languages == ["en"]
    assert info.bylaw_last_verified_at is None
