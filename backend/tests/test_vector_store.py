"""Unit tests for app.services.vector_store write path (Phase 2, Step 3).

Hermetic: the Supabase client is replaced by a stub that records the calls
it receives, so these assert the request shape and the two invariants the
module exists to hold - idempotent upsert on the natural key, and removal
of chunks a re-ingestion orphaned.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.services.chunker import ChunkPayload
from app.services.vector_store import (
    NATURAL_KEY,
    TABLE,
    BylawChunkStore,
    chunk_fingerprint,
)

DIMENSIONS = 1024


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        supabase_url="https://test-placeholder.supabase.co",
        supabase_service_role_key="test-placeholder",
    )


def payload(section: str = "8.14(4)", index: int = 0) -> ChunkPayload:
    return ChunkPayload(
        municipality_id="nb_fredericton",
        province_code="NB",
        bylaw_name="Zoning By-law Z-5",
        section_number=section,
        section_title="Standards",
        chunk_content="[Province: NB] ... Section 8.14(4):\n(a) Lot Area (MIN)",
        language="en",
        page_number=167,
        source_document_version="2023-04",
        metadata={"section": section},
        chunk_index=index,
    )


class StubQuery:
    """Records one PostgREST call chain."""

    def __init__(self, table: "StubTable") -> None:
        self._table = table
        self._filters: dict[str, str] = {}

    def upsert(self, rows, on_conflict=None):
        self._table.upserts.append({"rows": rows, "on_conflict": on_conflict})
        return self

    def select(self, columns, count=None):
        self._table.selects.append(columns)
        return self

    def delete(self):
        self._table.deleted_called = True
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def in_(self, column, values):
        self._table.deleted_ids.extend(values)
        return self

    @property
    def not_(self):
        return self

    def is_(self, column, value):
        self._table.not_is_filters.append((column, value))
        return self

    def range(self, start, end):
        self._table.ranges.append((start, end))
        return self

    async def execute(self):
        return StubResponse(self._table.existing_rows, count=len(self._table.existing_rows))


class StubResponse:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class StubTable:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.selects: list[str] = []
        self.deleted_ids: list[str] = []
        self.deleted_called = False
        self.existing_rows: list[dict] = []
        self.not_is_filters: list[tuple] = []
        self.ranges: list[tuple] = []


class StubRpc:
    def __init__(self, client: "StubClient", name: str, params: dict) -> None:
        self._client = client
        client.rpc_calls.append({"name": name, "params": params})

    async def execute(self):
        return StubResponse(self._client.rpc_rows)


class StubClient:
    def __init__(self) -> None:
        self.tables: dict[str, StubTable] = {}
        self.rpc_calls: list[dict] = []
        self.rpc_rows: list[dict] = []

    def table(self, name):
        self.tables.setdefault(name, StubTable())
        return StubQuery(self.tables[name])

    def rpc(self, name, params):
        return StubRpc(self, name, params)


def make_store() -> tuple[BylawChunkStore, StubClient]:
    client = StubClient()
    return BylawChunkStore(client, settings()), client  # type: ignore[arg-type]


def vector(value: float = 0.1) -> list[float]:
    return [value] * DIMENSIONS


# ---------------------------------------------------------------------
#  Upsert
# ---------------------------------------------------------------------


def test_upsert_targets_the_natural_key():
    """Without on_conflict, re-ingestion doubles the corpus."""
    store, client = make_store()
    asyncio.run(store.upsert_chunks([payload()], [vector()]))

    call = client.tables[TABLE].upserts[0]
    assert call["on_conflict"] == NATURAL_KEY
    assert NATURAL_KEY == "municipality_id,bylaw_name,section_number,language,chunk_index"


def test_upsert_attaches_each_embedding_to_its_own_row():
    store, client = make_store()
    payloads = [payload("8.1(1)", 0), payload("8.2(1)", 0)]
    embeddings = [vector(0.1), vector(0.2)]
    asyncio.run(store.upsert_chunks(payloads, embeddings))

    rows = client.tables[TABLE].upserts[0]["rows"]
    assert rows[0]["section_number"] == "8.1(1)"
    assert rows[0]["embedding"][0] == pytest.approx(0.1)
    assert rows[1]["section_number"] == "8.2(1)"
    assert rows[1]["embedding"][0] == pytest.approx(0.2)


def test_length_mismatch_is_refused_rather_than_paired_by_position():
    store, _ = make_store()
    with pytest.raises(ValueError, match="refusing to pair"):
        asyncio.run(store.upsert_chunks([payload(), payload()], [vector()]))


def test_empty_input_makes_no_request():
    store, client = make_store()
    assert asyncio.run(store.upsert_chunks([], [])) == 0
    assert TABLE not in client.tables


def test_rows_are_batched():
    store, client = make_store()
    payloads = [payload(f"8.{i}(1)") for i in range(250)]
    written = asyncio.run(store.upsert_chunks(payloads, [vector()] * 250, batch_size=100))

    assert written == 250
    sizes = [len(call["rows"]) for call in client.tables[TABLE].upserts]
    assert sizes == [100, 100, 50]


def test_row_carries_every_table_column():
    store, client = make_store()
    asyncio.run(store.upsert_chunks([payload()], [vector()]))
    row = client.tables[TABLE].upserts[0]["rows"][0]
    assert row["municipality_id"] == "nb_fredericton"
    assert row["province_code"] == "NB"
    assert row["language"] == "en"
    assert row["chunk_index"] == 0
    assert len(row["embedding"]) == DIMENSIONS


# ---------------------------------------------------------------------
#  Orphan removal
# ---------------------------------------------------------------------


def test_orphaned_chunks_are_deleted():
    """An amended clause needing fewer chunks leaves repealed text indexed."""
    store, client = make_store()
    client.tables.setdefault(TABLE, StubTable())
    client.tables[TABLE].existing_rows = [
        {"id": "keep-1", "section_number": "8.14(4)", "chunk_index": 0},
        {"id": "stale-1", "section_number": "8.14(4)", "chunk_index": 1},
        {"id": "stale-2", "section_number": "8.99(9)", "chunk_index": 0},
    ]

    deleted = asyncio.run(
        store.delete_orphaned_chunks(
            "nb_fredericton", "Zoning By-law Z-5", "en", [payload("8.14(4)", 0)]
        )
    )

    assert deleted == 2
    assert sorted(client.tables[TABLE].deleted_ids) == ["stale-1", "stale-2"]


def test_nothing_is_deleted_when_the_document_is_unchanged():
    store, client = make_store()
    client.tables.setdefault(TABLE, StubTable())
    client.tables[TABLE].existing_rows = [
        {"id": "keep-1", "section_number": "8.14(4)", "chunk_index": 0},
    ]

    deleted = asyncio.run(
        store.delete_orphaned_chunks(
            "nb_fredericton", "Zoning By-law Z-5", "en", [payload("8.14(4)", 0)]
        )
    )
    assert deleted == 0
    assert client.tables[TABLE].deleted_ids == []


def test_replace_document_upserts_before_deleting():
    """Delete-first would leave the municipality with no bylaw if the run fails."""
    store, client = make_store()
    client.tables.setdefault(TABLE, StubTable())
    client.tables[TABLE].existing_rows = [
        {"id": "stale-1", "section_number": "old", "chunk_index": 0},
    ]

    result = asyncio.run(store.replace_document([payload()], [vector()]))

    assert result.upserted == 1
    assert result.deleted == 1
    assert client.tables[TABLE].upserts, "upsert must happen before any delete"


def test_replace_document_on_empty_input_is_a_no_op():
    store, client = make_store()
    result = asyncio.run(store.replace_document([], []))
    assert (result.upserted, result.deleted) == (0, 0)
    assert TABLE not in client.tables


# ---------------------------------------------------------------------
#  Resume support
#
#  Under a capped daily quota an all-or-nothing run never converges: it
#  spends quota, dies partway, discards the work, and spends the same
#  quota again next time. Resuming needs to know what is already done.
# ---------------------------------------------------------------------


def test_embedded_chunks_are_fingerprinted_by_their_text():
    store, client = make_store()
    client.tables.setdefault(TABLE, StubTable())
    client.tables[TABLE].existing_rows = [
        {"section_number": "8.14(4)", "chunk_index": 0, "chunk_content": "a"},
        {"section_number": "8.14(4)", "chunk_index": 1, "chunk_content": "b"},
        {"section_number": "3(85)", "chunk_index": 0, "chunk_content": "a"},
    ]

    done = asyncio.run(
        store.embedded_chunk_fingerprints(
            "nb_fredericton", "Zoning By-law Z-5", "fr"
        )
    )
    assert set(done) == {("8.14(4)", 0), ("8.14(4)", 1), ("3(85)", 0)}
    # Same text, same fingerprint; different text, different fingerprint.
    assert done[("8.14(4)", 0)] == done[("3(85)", 0)]
    assert done[("8.14(4)", 1)] != done[("8.14(4)", 0)]


def test_a_rewritten_chunk_is_not_treated_as_done():
    """A parser fix changes what a clause says, not where it sits.

    Resuming on the natural key alone would skip those rows for good, and
    Fredericton's sign matrix would still name its zones "P" and "DA".
    """
    store, client = make_store()
    client.tables.setdefault(TABLE, StubTable())
    client.tables[TABLE].existing_rows = [
        {
            "section_number": "6.4",
            "chunk_index": 0,
            "chunk_content": "CANOPY 6.4(1) - Permitted: P, P.",
        }
    ]

    done = asyncio.run(store.embedded_chunk_fingerprints("nb_fredericton", "Z-5", "en"))
    corrected = "CANOPY 6.4(1) - Permitted: I-2, IEX, RT."
    assert done[("6.4", 0)] != chunk_fingerprint(corrected)


def test_embedded_fingerprints_are_empty_for_a_new_document():
    store, client = make_store()
    client.tables.setdefault(TABLE, StubTable())
    client.tables[TABLE].existing_rows = []
    assert asyncio.run(store.embedded_chunk_fingerprints("x", "y", "en")) == {}


def test_embedded_fingerprints_exclude_null_embeddings():
    """A row without a vector is unfinished work, not completed work."""
    store, client = make_store()
    table = client.tables.setdefault(TABLE, StubTable())
    table.existing_rows = [
        {"section_number": "8.1(1)", "chunk_index": 0, "chunk_content": "x"}
    ]

    asyncio.run(store.embedded_chunk_fingerprints("nb_fredericton", "Z-5", "en"))
    assert table.not_is_filters == [("embedding", "null")]


# ---------------------------------------------------------------------
#  Vector search (Phase 3, Step 1)
# ---------------------------------------------------------------------


def rpc_row(section: str = "8.14(2)", similarity: float = 0.748) -> dict:
    return {
        "id": "11111111-1111-4111-8111-111111111111",
        "municipality_id": "nb_fredericton",
        "province_code": "NB",
        "bylaw_name": "Zoning By-law Z-5",
        "section_number": section,
        "section_title": "Uses",
        "chunk_content": "(a) Permitted Uses\n(b) Conditional Uses\n(1) Kennel",
        "language": "en",
        "page_number": 167,
        "metadata": {"section": section},
        "similarity": similarity,
    }


def search(store, **kwargs):
    return asyncio.run(store.match_chunks(vector(0.1), "nb_fredericton", **kwargs))


def test_search_calls_the_spec_named_rpc():
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    search(store)
    assert client.rpc_calls[0]["name"] == "match_bylaw_chunks"


def test_municipality_is_filtered_inside_the_rpc():
    """Filtering after the RPC's LIMIT would let other municipalities win rows."""
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    search(store)
    assert client.rpc_calls[0]["params"]["target_municipality"] == "nb_fredericton"


def test_language_defaults_to_configured_rather_than_null():
    """NULL means "any language" in the RPC, breaking the bilingual lock."""
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    search(store)
    assert client.rpc_calls[0]["params"]["target_language"] == "en"


def test_language_can_be_overridden():
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    search(store, language="fr")
    assert client.rpc_calls[0]["params"]["target_language"] == "fr"


def test_threshold_and_count_default_from_settings():
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    search(store)
    params = client.rpc_calls[0]["params"]
    assert params["match_threshold"] == settings().match_threshold
    assert params["match_count"] == settings().vector_match_count


def test_threshold_and_count_can_be_overridden():
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    search(store, match_threshold=0.5, match_count=3)
    params = client.rpc_calls[0]["params"]
    assert params["match_threshold"] == 0.5
    assert params["match_count"] == 3


def test_rows_become_chunk_models():
    store, client = make_store()
    client.rpc_rows = [rpc_row()]
    hits = search(store)
    assert len(hits) == 1
    assert hits[0].section_number == "8.14(2)"
    assert hits[0].citation("Fredericton") == (
        "[Fredericton - Zoning By-law Z-5, Section 8.14(2)]"
    )


def test_score_is_seeded_from_similarity():
    """So a caller that skips the Step 2 merge still has a ranking value."""
    store, client = make_store()
    client.rpc_rows = [rpc_row(similarity=0.748)]
    assert search(store)[0].score == pytest.approx(0.748)


def test_no_matches_returns_an_empty_list():
    store, client = make_store()
    client.rpc_rows = []
    assert search(store) == []


def test_wrong_dimension_query_is_refused_before_the_call():
    store, client = make_store()
    with pytest.raises(ValueError, match="different models"):
        asyncio.run(store.match_chunks([0.1] * 768, "nb_fredericton"))
    assert client.rpc_calls == []
