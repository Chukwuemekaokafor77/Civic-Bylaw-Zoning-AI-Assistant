"""Supabase client for `bylaw_chunks` (Phase 2 writes, Phase 3 reads).

Phase 2 needs writes: upserting embedded chunks and clearing rows a
re-ingestion has orphaned. Phase 3 Step 1 adds `match_chunks`, the
`match_bylaw_chunks` RPC executor named in Section 6.

The service-role key is used here, which bypasses Row Level Security. That
is correct for an offline ingestion job writing to a table the public may
only read, and it is why this module must never be reachable from a
request handler that takes user input.

Two invariants this module exists to hold:

* Re-ingestion updates in place. Rows are upserted on the natural key
  (municipality, bylaw, section, language, chunk_index) from
  001_init_schema.sql, so running ingestion twice does not double the
  corpus - which would otherwise show up as the same rule retrieved
  repeatedly and cited as if from separate provisions.

* Orphans are deleted. If an amended bylaw splits a clause into three
  chunks where it previously needed five, the upsert refreshes indexes 0-2
  and leaves 3-4 holding repealed text - still indexed, still citable, and
  now wrong. `delete_orphaned_chunks` removes exactly those.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass

import structlog
from supabase import AsyncClient, create_async_client

from app.config import Settings, get_settings
from app.models.schemas import Chunk
from app.services.chunker import ChunkPayload

log = structlog.get_logger(__name__)

TABLE = "bylaw_chunks"

# The unique index from 001_init_schema.sql that makes the upsert idempotent.
NATURAL_KEY = "municipality_id,bylaw_name,section_number,language,chunk_index"

# Rows per request. Each carries a 1536-float embedding (~25 KB as JSON),
# so a larger batch risks the PostgREST body limit for no real gain.
UPSERT_BATCH_SIZE = 100


@dataclass
class UpsertResult:
    upserted: int
    deleted: int


def chunk_fingerprint(text: str) -> str:
    """Stable hash of a chunk's text, used to tell stale rows from done ones."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BylawChunkStore:
    """Write access to `bylaw_chunks`."""

    def __init__(self, client: AsyncClient, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "BylawChunkStore":
        settings = settings or get_settings()
        client = await create_async_client(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
        )
        return cls(client, settings)

    # -----------------------------------------------------------------
    #  Writes
    # -----------------------------------------------------------------

    async def upsert_chunks(
        self,
        payloads: list[ChunkPayload],
        embeddings: list[list[float]],
        *,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> int:
        """Upsert chunks with their embeddings, returning the row count."""
        if len(payloads) != len(embeddings):
            # Positional pairing is the only thing binding a vector to its
            # text. A length mismatch means the two lists have already
            # diverged, so refuse rather than write misaligned rows.
            raise ValueError(
                f"{len(payloads)} chunks but {len(embeddings)} embeddings; "
                "refusing to pair them by position."
            )
        if not payloads:
            return 0

        rows = []
        for payload, embedding in zip(payloads, embeddings):
            row = payload.as_row()
            row["embedding"] = embedding
            rows.append(row)

        written = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            await (
                self._client.table(TABLE)
                .upsert(batch, on_conflict=NATURAL_KEY)
                .execute()
            )
            written += len(batch)
            log.debug(
                "chunks_upserted",
                batch=f"{start // batch_size + 1}",
                rows=len(batch),
            )

        log.info(
            "chunk_upsert_complete",
            municipality=payloads[0].municipality_id,
            bylaw=payloads[0].bylaw_name,
            language=payloads[0].language,
            rows=written,
        )
        return written

    async def embedded_chunk_fingerprints(
        self,
        municipality_id: str,
        bylaw_name: str,
        language: str,
    ) -> dict[tuple[str, int], str]:
        """This document's embedded chunks, keyed by natural key.

        Lets an interrupted ingestion resume instead of restarting. Under a
        capped daily quota that is the difference between converging and
        never finishing: without it, a run that dies partway spends quota
        on embeddings it then discards, and the next run spends it again on
        exactly the same chunks.

        The value is a hash of the stored text, because the natural key
        alone cannot tell finished work from stale work. A parser fix
        rewrites what a clause says while its section number and index stay
        put, and skipping on the key would leave the old text embedded for
        good: Fredericton's sign matrix named its zones "P" and "DA", and a
        re-ingestion would have skipped every one of those rows.

        Rows with a NULL embedding are excluded - they are unfinished work,
        not completed work.
        """
        rows: list[dict] = []
        page_size = 1000
        offset = 0

        while True:
            response = (
                await self._client.table(TABLE)
                .select("section_number,chunk_index,chunk_content")
                .eq("municipality_id", municipality_id)
                .eq("bylaw_name", bylaw_name)
                .eq("language", language)
                .not_.is_("embedding", "null")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = response.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        return {
            (row["section_number"], row["chunk_index"]): chunk_fingerprint(
                row.get("chunk_content") or ""
            )
            for row in rows
        }

    async def delete_orphaned_chunks(
        self,
        municipality_id: str,
        bylaw_name: str,
        language: str,
        keep: list[ChunkPayload],
    ) -> int:
        """Delete rows for this document that the current run did not write.

        Computed by difference against what is actually in the table rather
        than by a blanket delete-then-insert: a delete-first strategy leaves
        the municipality with no indexed bylaw at all if the run then fails,
        and this job is the only thing standing between the public and an
        empty answer.
        """
        current = {(p.section_number, p.chunk_index) for p in keep}

        existing = (
            await self._client.table(TABLE)
            .select("id,section_number,chunk_index")
            .eq("municipality_id", municipality_id)
            .eq("bylaw_name", bylaw_name)
            .eq("language", language)
            .execute()
        )

        orphans = [
            row["id"]
            for row in existing.data or []
            if (row["section_number"], row["chunk_index"]) not in current
        ]
        if not orphans:
            return 0

        for start in range(0, len(orphans), UPSERT_BATCH_SIZE):
            batch = orphans[start : start + UPSERT_BATCH_SIZE]
            await self._client.table(TABLE).delete().in_("id", batch).execute()

        log.info(
            "orphaned_chunks_deleted",
            municipality=municipality_id,
            bylaw=bylaw_name,
            language=language,
            rows=len(orphans),
        )
        return len(orphans)

    async def replace_document(
        self,
        payloads: list[ChunkPayload],
        embeddings: list[list[float]],
    ) -> UpsertResult:
        """Upsert a document's chunks, then clear anything left behind."""
        if not payloads:
            return UpsertResult(upserted=0, deleted=0)

        first = payloads[0]
        upserted = await self.upsert_chunks(payloads, embeddings)
        deleted = await self.delete_orphaned_chunks(
            first.municipality_id,
            first.bylaw_name,
            first.language,
            payloads,
        )
        return UpsertResult(upserted=upserted, deleted=deleted)

    # -----------------------------------------------------------------
    #  Reads (Phase 3, Step 1)
    # -----------------------------------------------------------------

    async def match_chunks(
        self,
        query_embedding: list[float],
        municipality_id: str,
        *,
        language: str | None = None,
        match_threshold: float | None = None,
        match_count: int | None = None,
    ) -> list[Chunk]:
        """Vector search via the `match_bylaw_chunks` RPC.

        `municipality_id` is passed to the RPC, not applied afterwards.
        Filtering in Python would mean the LIMIT inside the function had
        already chosen its rows from every municipality in the table, so a
        Fredericton question could come back with five Saint John clauses
        and then be trimmed to nothing - or worse, to a few that slipped
        through. Section 5 Rule 2 forbids mixing municipalities, and this
        is where that is actually enforced.

        `language` defaults to the configured default rather than NULL for
        the same reason: NULL means "any language" in the RPC, which under
        the bilingual lock would let a French chunk be cited in an English
        answer.
        """
        if len(query_embedding) != self._settings.embedding_dimensions:
            raise ValueError(
                f"Query embedding has {len(query_embedding)} dimensions, "
                f"expected {self._settings.embedding_dimensions}. The RPC "
                "would reject it, but a mismatch here means query and "
                "document vectors came from different models."
            )

        params = {
            "query_embedding": query_embedding,
            "target_municipality": municipality_id,
            "target_language": language or self._settings.default_language,
            "match_threshold": (
                self._settings.match_threshold if match_threshold is None else match_threshold
            ),
            "match_count": (
                self._settings.vector_match_count if match_count is None else match_count
            ),
        }

        response = await self._client.rpc("match_bylaw_chunks", params).execute()
        rows = response.data or []

        chunks = [Chunk.model_validate(row) for row in rows]
        for chunk in chunks:
            # Seed the fused score with the vector score so a caller that
            # skips the Step 2 merge still has something to rank on.
            chunk.score = chunk.similarity

        log.info(
            "vector_search",
            municipality=municipality_id,
            language=params["target_language"],
            threshold=params["match_threshold"],
            requested=params["match_count"],
            returned=len(chunks),
            top_similarity=round(chunks[0].similarity, 4) if chunks and chunks[0].similarity else None,
        )
        return chunks

    async def count_chunks(self, municipality_id: str, language: str | None = None) -> int:
        """Row count for a municipality - used by the Step 5 run summary."""
        query = (
            self._client.table(TABLE)
            .select("id", count="exact")
            .eq("municipality_id", municipality_id)
        )
        if language:
            query = query.eq("language", language)
        response = await query.execute()
        return response.count or 0
