"""Unit tests for app.services.audit_logger (Phase 3, Step 4)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app.config import Settings
from app.models.schemas import Chunk
from app.services.audit_logger import MAX_RESPONSE_CHARS, TABLE, AuditLogger, QueryRecord


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        supabase_url="https://test-placeholder.supabase.co",
        supabase_service_role_key="test-placeholder",
    )


class StubQuery:
    def __init__(self, table: "StubTable") -> None:
        self._table = table

    def insert(self, row):
        if self._table.raise_on_write:
            raise RuntimeError("supabase unreachable")
        self._table.inserts.append(row)
        return self

    def select(self, columns, count=None):
        return self

    def eq(self, column, value):
        self._table.filters.append((column, value))
        return self

    def limit(self, n):
        return self

    async def execute(self):
        return type("R", (), {"data": [], "count": self._table.count})()


class StubTable:
    def __init__(self) -> None:
        self.inserts: list[dict] = []
        self.filters: list[tuple] = []
        self.count = 0
        self.raise_on_write = False


class StubClient:
    def __init__(self) -> None:
        self.tables: dict[str, StubTable] = {}

    def table(self, name):
        self.tables.setdefault(name, StubTable())
        return StubQuery(self.tables[name])


def make_logger() -> tuple[AuditLogger, StubClient]:
    client = StubClient()
    return AuditLogger(client, settings()), client  # type: ignore[arg-type]


def chunk() -> Chunk:
    return Chunk(
        id=uuid4(),
        municipality_id="nb_fredericton",
        province_code="NB",
        bylaw_name="Zoning By-law Z-5",
        section_number="8.14(2)",
        chunk_content="body",
    )


def record(**overrides) -> QueryRecord:
    base = dict(
        municipality_id="nb_fredericton",
        user_query="Are kennels allowed?",
        retrieved_chunk_ids=[uuid4()],
        response_text="Kennels are a conditional use.",
        language="en",
    )
    base.update(overrides)
    return QueryRecord(**base)


# ---------------------------------------------------------------------
#  Row shape
# ---------------------------------------------------------------------


def test_row_matches_the_query_log_columns():
    assert set(record().as_row()) == {
        "municipality_id",
        "user_query",
        "retrieved_chunk_ids",
        "response_text",
        "was_fallback",
        "language",
    }


def test_session_id_is_never_stored():
    """ChatRequest.session_id is for rate limiting, not identification."""
    assert "session_id" not in record().as_row()


def test_chunk_ids_are_serialised_as_strings():
    row = record().as_row()
    assert all(isinstance(cid, str) for cid in row["retrieved_chunk_ids"])


def test_oversized_response_is_truncated():
    row = record(response_text="x" * (MAX_RESPONSE_CHARS + 500)).as_row()
    assert len(row["response_text"]) < MAX_RESPONSE_CHARS + 100
    assert row["response_text"].endswith("[truncated]")


def test_normal_response_is_stored_intact():
    text = "Kennels are a **conditional use** [Fredericton - Z-5, Section 8.14(2)]."
    assert record(response_text=text).as_row()["response_text"] == text


# ---------------------------------------------------------------------
#  Fallback derivation
# ---------------------------------------------------------------------


def test_no_retrieved_chunks_is_recorded_as_a_fallback():
    """The coverage-gap signal Section 7 asks for."""
    built = QueryRecord.from_answer(
        municipality_id="nb_fredericton",
        user_query="parking on Mars",
        chunks=[],
        response_text="I could not find a specific rule...",
    )
    assert built.was_fallback is True
    assert built.retrieved_chunk_ids == []


def test_retrieved_chunks_are_not_a_fallback():
    built = QueryRecord.from_answer(
        municipality_id="nb_fredericton",
        user_query="Are kennels allowed?",
        chunks=[chunk()],
        response_text="Kennels are conditional.",
    )
    assert built.was_fallback is False
    assert len(built.retrieved_chunk_ids) == 1


def test_explicit_fallback_flag_overrides_the_default():
    built = QueryRecord.from_answer(
        municipality_id="nb_fredericton",
        user_query="q",
        chunks=[chunk()],
        response_text="I could not find a specific rule...",
        was_fallback=True,
    )
    assert built.was_fallback is True


def test_language_is_carried_through():
    built = QueryRecord.from_answer(
        municipality_id="nb_fredericton",
        user_query="Les chenils sont-ils permis?",
        chunks=[chunk()],
        response_text="...",
        language="fr",
    )
    assert built.as_row()["language"] == "fr"


# ---------------------------------------------------------------------
#  Writing
# ---------------------------------------------------------------------


def test_record_inserts_into_query_log():
    logger, client = make_logger()
    assert asyncio.run(logger.record(record())) is True
    assert len(client.tables[TABLE].inserts) == 1


def test_a_failed_write_never_breaks_the_answer():
    """The response has already reached the user; raising helps nobody."""
    logger, client = make_logger()
    client.tables.setdefault(TABLE, StubTable()).raise_on_write = True

    assert asyncio.run(logger.record(record())) is False


def test_fallback_rate_counts_both_totals():
    logger, client = make_logger()
    client.tables.setdefault(TABLE, StubTable()).count = 7
    fallbacks, total = asyncio.run(logger.fallback_rate("nb_fredericton"))
    assert (fallbacks, total) == (7, 7)


def test_fallback_rate_filters_on_the_fallback_flag():
    logger, client = make_logger()
    client.tables.setdefault(TABLE, StubTable())
    asyncio.run(logger.fallback_rate("nb_fredericton"))
    assert ("was_fallback", True) in client.tables[TABLE].filters
