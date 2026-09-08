"""Unit tests for app.services.embedder (Phase 2, Step 1).

Hermetic: HTTP goes through httpx.MockTransport - a real client over fake
responses, so batching, retries and response parsing are genuinely
exercised - and Settings is constructed explicitly so CI needs no
VOYAGE_API_KEY.

Tests use asyncio.run rather than pytest-asyncio markers because the repo
has no asyncio_mode configuration; this keeps them runnable under a bare
`pytest -q tests`.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.services.embedder import (
    MAX_CHARS_PER_INPUT,
    TASK_DOCUMENT,
    TASK_QUERY,
    Embedder,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingInputTooLarge,
    EmbeddingQuotaExhausted,
    estimate_tokens,
)

DIMENSIONS = 1024

_REAL_SLEEP = asyncio.sleep


def _instant_sleep(_seconds: float):
    """Drop-in for asyncio.sleep so backoff does not slow the suite."""
    return _REAL_SLEEP(0)


def make_settings(**overrides) -> Settings:
    base = {
        "supabase_url": "https://test-placeholder.supabase.co",
        "supabase_service_role_key": "test-placeholder",
        "voyage_api_key": "test-not-real",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def vector(first: float) -> list[float]:
    """A distinctly identifiable vector of the right width."""
    return [first] + [0.0] * (DIMENSIONS - 1)


class Recorder:
    """Captures the requests a handler received."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    @property
    def calls(self) -> int:
        return len(self.payloads)


def ok_handler(recorder: Recorder, dimensions: int = DIMENSIONS, reverse: bool = True):
    """Well-formed responses, deliberately out of index order by default."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        recorder.payloads.append(payload)
        count = len(payload["input"])
        data = [
            {"index": i, "embedding": [float(i + 1)] + [0.0] * (dimensions - 1)}
            for i in range(count)
        ]
        if reverse:
            data.reverse()
        return httpx.Response(
            200, json={"data": data, "usage": {"total_tokens": count * 10}}
        )

    return handler


def make_embedder(handler, **overrides) -> Embedder:
    embedder = Embedder(make_settings(**overrides))
    embedder._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test"},
    )
    return embedder


# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------


def test_missing_api_key_raises_config_error():
    with pytest.raises(EmbeddingConfigError, match="VOYAGE_API_KEY"):
        Embedder(make_settings(voyage_api_key=None))


def test_config_error_points_at_the_free_grant():
    with pytest.raises(EmbeddingConfigError, match="voyageai.com"):
        Embedder(make_settings(voyage_api_key=None))


def test_dimensions_must_match_the_column():
    """A mismatch fails at insert with an opaque pgvector error."""
    with pytest.raises(ValueError, match="1024"):
        make_settings(embedding_dimensions=1536)


# ---------------------------------------------------------------------
#  Request shape
# ---------------------------------------------------------------------


def test_model_and_dimension_are_sent_explicitly():
    """The model default could change; VECTOR(1024) cannot."""
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    asyncio.run(embedder.embed_documents(["a"]))

    assert recorder.payloads[0]["model"] == "voyage-4-large"
    assert recorder.payloads[0]["output_dimension"] == DIMENSIONS


def test_documents_use_the_document_input_type():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    asyncio.run(embedder.embed_documents(["clause text"]))
    assert recorder.payloads[0]["input_type"] == TASK_DOCUMENT


def test_queries_use_the_query_input_type():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    asyncio.run(embedder.embed_query("can I build a garden suite?"))
    assert recorder.payloads[0]["input_type"] == TASK_QUERY


def test_query_is_not_given_the_section_4_context_prefix():
    """Both RPCs already filter by municipality; prefixing wastes similarity."""
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    asyncio.run(embedder.embed_query("  what does section 6.3 say?  "))
    assert recorder.payloads[0]["input"] == ["what does section 6.3 say?"]


# ---------------------------------------------------------------------
#  Ordering and response validation
# ---------------------------------------------------------------------


def test_vectors_are_sorted_back_into_request_order():
    """The API returns an index and does not guarantee response order."""
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    vectors = asyncio.run(embedder.embed_documents(["a", "b", "c", "d"]))
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0]


def test_every_vector_has_the_column_width():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    vectors = asyncio.run(embedder.embed_documents(["a", "b"]))
    assert all(len(v) == DIMENSIONS for v in vectors)


def test_wrong_dimension_output_is_caught_before_insert():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder, dimensions=768))
    with pytest.raises(EmbeddingError, match="768 dimensions"):
        asyncio.run(embedder.embed_documents(["a"]))


def test_count_mismatch_refuses_positional_alignment():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": vector(1.0)}], "usage": {}},
        )

    embedder = make_embedder(handler)
    with pytest.raises(EmbeddingError, match="refusing to align"):
        asyncio.run(embedder.embed_documents(["a", "b"]))


def test_empty_input_list_makes_no_request():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    assert asyncio.run(embedder.embed_documents([])) == []
    assert recorder.calls == 0


def test_embed_query_returns_a_single_vector():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    assert len(asyncio.run(embedder.embed_query("a question"))) == DIMENSIONS


# ---------------------------------------------------------------------
#  Batching
# ---------------------------------------------------------------------


def test_batches_respect_the_input_count_ceiling():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    asyncio.run(embedder.embed_documents(["x"] * 250, batch_size=96))
    assert [len(p["input"]) for p in recorder.payloads] == [96, 96, 58]


def test_batches_split_on_the_token_budget():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    asyncio.run(
        embedder.embed_documents(["y" * 10_000] * 5, batch_size=96, batch_token_budget=6_000)
    )
    assert all(len(p["input"]) <= 2 for p in recorder.payloads)


def test_batching_never_drops_or_reorders_inputs():
    embedder = make_embedder(ok_handler(Recorder()))
    texts = [f"clause {i}" for i in range(50)]
    flat = [t for batch in embedder._make_batches(texts, 7, 10**9) for t in batch]
    assert flat == texts


def test_multiple_batches_concatenate_in_order():
    recorder = Recorder()
    embedder = make_embedder(ok_handler(recorder))
    vectors = asyncio.run(embedder.embed_documents(["a", "b", "c", "d"], batch_size=2))
    assert recorder.calls == 2
    assert len(vectors) == 4


# ---------------------------------------------------------------------
#  Input validation
# ---------------------------------------------------------------------


def test_blank_input_is_rejected_with_its_index():
    embedder = make_embedder(ok_handler(Recorder()))
    with pytest.raises(EmbeddingError, match="index 1 is empty"):
        embedder._validate_inputs(["fine", "   "])


def test_oversized_input_raises_rather_than_truncating():
    """Truncation would drop bylaw text while still yielding a citable chunk."""
    embedder = make_embedder(ok_handler(Recorder()))
    with pytest.raises(EmbeddingInputTooLarge):
        embedder._validate_inputs(["z" * (MAX_CHARS_PER_INPUT + 1)])


def test_empty_query_is_rejected():
    embedder = make_embedder(ok_handler(Recorder()))
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed_query("   "))


# ---------------------------------------------------------------------
#  Failure handling
# ---------------------------------------------------------------------


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 3:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": vector(1.0)}], "usage": {}}
        )

    embedder = make_embedder(handler)
    assert len(asyncio.run(embedder.embed_documents(["a"]))) == 1
    assert state["calls"] == 3


def test_server_error_is_retried(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": vector(1.0)}], "usage": {}}
        )

    embedder = make_embedder(handler)
    asyncio.run(embedder.embed_documents(["a"]))
    assert state["calls"] == 2


def test_bad_request_fails_on_the_first_attempt(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(400, text="malformed")

    embedder = make_embedder(handler)
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed_documents(["a"]))
    assert state["calls"] == 1


def test_exhausted_grant_stops_immediately(monkeypatch):
    """A spent grant does not refill; retrying only delays the diagnosis."""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(402, text="insufficient quota on this account")

    embedder = make_embedder(handler)
    with pytest.raises(EmbeddingQuotaExhausted):
        asyncio.run(embedder.embed_documents(["a"]))
    assert state["calls"] == 1


def test_quota_error_is_an_embedding_error_subclass():
    """Callers that catch EmbeddingError broadly still handle it."""
    assert issubclass(EmbeddingQuotaExhausted, EmbeddingError)


# ---------------------------------------------------------------------
#  Token estimation
# ---------------------------------------------------------------------


def test_token_estimate_is_conservative():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 3500) >= 1000
