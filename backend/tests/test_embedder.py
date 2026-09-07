"""Unit tests for app.services.embedder (Phase 2, Step 1).

Hermetic: no network, no API key required. The Gemini client is replaced
with a stub, and Settings is constructed explicitly rather than read from
the environment so CI needs no GEMINI_API_KEY.

Tests use asyncio.run rather than pytest-asyncio markers because the repo
has no asyncio_mode configuration; this keeps them runnable under a bare
`pytest -q tests`.
"""

from __future__ import annotations

import asyncio
import math
import types

import pytest
from google.genai.errors import ClientError, ServerError

from app.config import Settings
from app.services.embedder import (
    MAX_CHARS_PER_INPUT,
    MAX_INPUTS_PER_REQUEST,
    TASK_DOCUMENT,
    TASK_QUERY,
    Embedder,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingInputTooLarge,
    EmbeddingQuotaExhausted,
    _is_retryable,
    _RateLimiter,
    estimate_tokens,
    is_daily_quota,
    normalize,
    retry_delay_from,
)

DIMENSIONS = 1536

# Captured before any monkeypatching so the no-op replacement below does
# not recurse into itself.
_REAL_SLEEP = asyncio.sleep


def _instant_sleep(_seconds: float):
    """Drop-in for asyncio.sleep so backoff does not slow the suite."""
    return _REAL_SLEEP(0)


def make_settings(**overrides) -> Settings:
    base = {
        "supabase_url": "https://test-placeholder.supabase.co",
        "supabase_service_role_key": "test-placeholder",
        "gemini_api_key": "test-not-real",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# One Embedder is shared across tests; each test still supplies its own
# stub client. Construction is cheap here, but keeping the pattern means
# the suite does not depend on how costly the SDK client is to build.
_SHARED: Embedder | None = None


def make_embedder(stub=None) -> Embedder:
    global _SHARED
    if _SHARED is None:
        _SHARED = Embedder(make_settings())
    # The SDK nests the async surface at client.aio.models.embed_content.
    _SHARED._client = (
        types.SimpleNamespace(aio=types.SimpleNamespace(models=stub)) if stub else None
    )
    return _SHARED


def api_error(cls, code: int):
    """Build an SDK error without going through its HTTP response parsing."""
    err = cls.__new__(cls)
    Exception.__init__(err, f"stub {code}")
    err.code = code
    err.message = f"stub {code}"
    return err


def embed_response(vectors):
    return types.SimpleNamespace(
        embeddings=[types.SimpleNamespace(values=v) for v in vectors]
    )


def vector_of(first: float) -> list[float]:
    """A distinctly-identifiable, deliberately non-unit vector."""
    return [first] + [0.0] * (DIMENSIONS - 1)


class StubEmbeddings:
    """Returns one well-formed embedding per input, in request order."""

    def __init__(self) -> None:
        self.calls = 0
        self.tasks: list[str] = []
        self.dimensions: list[int] = []
        self.batches: list[list[str]] = []

    async def embed_content(self, *, model, contents, config):
        self.calls += 1
        self.tasks.append(config.task_type)
        self.dimensions.append(config.output_dimensionality)
        self.batches.append(list(contents))
        return embed_response([vector_of(float(i + 1)) for i in range(len(contents))])


# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------


def test_missing_api_key_raises_config_error():
    with pytest.raises(EmbeddingConfigError, match="GEMINI_API_KEY"):
        Embedder(make_settings(gemini_api_key=None))


def test_config_error_points_at_the_free_key_page():
    with pytest.raises(EmbeddingConfigError, match="aistudio.google.com"):
        Embedder(make_settings(gemini_api_key=None))


def test_requested_dimensionality_matches_the_column():
    """The model defaults to 3072, which would not fit VECTOR(1536)."""
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    asyncio.run(embedder.embed_documents(["a"]))
    assert stub.dimensions == [DIMENSIONS]


# ---------------------------------------------------------------------
#  Task types — asymmetric retrieval
# ---------------------------------------------------------------------


def test_documents_are_embedded_as_retrieval_documents():
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    asyncio.run(embedder.embed_documents(["clause text"]))
    assert stub.tasks == [TASK_DOCUMENT]


def test_queries_are_embedded_as_retrieval_queries():
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    asyncio.run(embedder.embed_query("can I build a garden suite?"))
    assert stub.tasks == [TASK_QUERY]


def test_query_is_not_given_the_section_4_context_prefix():
    """Both RPCs already filter by municipality; prefixing wastes similarity."""
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    asyncio.run(embedder.embed_query("  what does section 6.3 say?  "))
    assert stub.batches == [["what does section 6.3 say?"]]


# ---------------------------------------------------------------------
#  Normalisation
# ---------------------------------------------------------------------


def test_normalize_returns_unit_length():
    assert math.isclose(sum(v * v for v in normalize([3.0, 4.0])), 1.0)


def test_normalize_leaves_a_zero_vector_alone():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_returned_vectors_are_normalised():
    """Truncated Matryoshka output from this model is not unit length."""
    embedder = make_embedder(StubEmbeddings())
    vectors = asyncio.run(embedder.embed_documents(["a", "b"]))
    for vector in vectors:
        assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------
#  Batching
# ---------------------------------------------------------------------


def test_batches_respect_input_count_ceiling():
    embedder = make_embedder()
    batches = embedder._make_batches(["x" * 100] * 150, 64, 200_000)
    assert [len(b) for b in batches] == [64, 64, 22]


def test_batches_split_on_token_budget():
    embedder = make_embedder()
    # ~2857 estimated tokens each against a 6000-token budget.
    batches = embedder._make_batches(["y" * 10_000] * 5, 64, 6_000)
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == 5


def test_batching_never_drops_or_reorders_inputs():
    embedder = make_embedder()
    texts = [f"clause {i}" for i in range(50)]
    flat = [t for batch in embedder._make_batches(texts, 7, 10**9) for t in batch]
    assert flat == texts


def test_batch_size_is_clamped_to_provider_limit():
    embedder = make_embedder()
    batches = embedder._make_batches(["x"] * 3000, 99_999, 10**9)
    assert max(len(b) for b in batches) <= MAX_INPUTS_PER_REQUEST


def test_multiple_batches_are_concatenated_in_order():
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    vectors = asyncio.run(embedder.embed_documents(["a", "b", "c", "d"], batch_size=2))
    assert stub.calls == 2
    assert [v[0] for v in vectors] == [1.0, 1.0, 1.0, 1.0]
    assert len(vectors) == 4


# ---------------------------------------------------------------------
#  Input validation
# ---------------------------------------------------------------------


def test_blank_input_is_rejected_with_its_index():
    embedder = make_embedder()
    with pytest.raises(EmbeddingError, match="index 1 is empty"):
        embedder._validate_inputs(["fine", "   "])


def test_oversized_input_raises_rather_than_truncating():
    embedder = make_embedder()
    with pytest.raises(EmbeddingInputTooLarge):
        embedder._validate_inputs(["z" * (MAX_CHARS_PER_INPUT + 1)])


def test_empty_query_is_rejected():
    embedder = make_embedder(StubEmbeddings())
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed_query("   "))


# ---------------------------------------------------------------------
#  Ordering and response validation
# ---------------------------------------------------------------------


def test_vectors_are_returned_in_request_order():
    """Gemini returns no per-item index, so order is positional."""
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    vectors = asyncio.run(embedder.embed_documents(["a", "b", "c", "d"]))
    assert [round(v[0], 6) for v in vectors] == [1.0, 1.0, 1.0, 1.0]
    assert all(len(v) == DIMENSIONS for v in vectors)


def test_empty_input_list_makes_no_call():
    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    assert asyncio.run(embedder.embed_documents([])) == []
    assert stub.calls == 0


def test_embed_query_returns_a_single_vector():
    embedder = make_embedder(StubEmbeddings())
    vector = asyncio.run(embedder.embed_query("what does section 6.3 say?"))
    assert len(vector) == DIMENSIONS


def test_dimension_mismatch_is_caught_before_insert():
    class WrongDims:
        async def embed_content(self, *, model, contents, config):
            return embed_response([[0.1] * 768])

    embedder = make_embedder(WrongDims())
    with pytest.raises(EmbeddingError, match="768 dimensions"):
        asyncio.run(embedder.embed_documents(["a"]))


def test_aggregated_response_is_refused():
    """gemini-embedding-2 collapses a list into ONE vector; that must not pass."""

    class Aggregating:
        async def embed_content(self, *, model, contents, config):
            return embed_response([vector_of(1.0)])

    embedder = make_embedder(Aggregating())
    with pytest.raises(EmbeddingError, match="refusing to align"):
        asyncio.run(embedder.embed_documents(["a", "b", "c"]))


def test_aggregation_error_names_the_likely_cause():
    class Aggregating:
        async def embed_content(self, *, model, contents, config):
            return embed_response([vector_of(1.0)])

    embedder = make_embedder(Aggregating())
    with pytest.raises(EmbeddingError, match="one embedding per input"):
        asyncio.run(embedder.embed_documents(["a", "b"]))


def test_missing_embeddings_field_is_refused():
    class Empty:
        async def embed_content(self, *, model, contents, config):
            return types.SimpleNamespace(embeddings=None)

    embedder = make_embedder(Empty())
    with pytest.raises(EmbeddingError, match="refusing to align"):
        asyncio.run(embedder.embed_documents(["a"]))


# ---------------------------------------------------------------------
#  Retry policy
# ---------------------------------------------------------------------


def test_server_error_is_retryable():
    assert _is_retryable(api_error(ServerError, 503)) is True


def test_rate_limit_is_retryable():
    """A 429 is an expected part of a large run on the free tier."""
    assert _is_retryable(api_error(ClientError, 429)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_other_client_errors_are_not_retryable(code):
    assert _is_retryable(api_error(ClientError, code)) is False


def test_unrelated_exception_is_not_retryable():
    assert _is_retryable(ValueError("unrelated")) is False


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    class FlakyThenOk:
        def __init__(self) -> None:
            self.calls = 0

        async def embed_content(self, *, model, contents, config):
            self.calls += 1
            if self.calls < 3:
                raise api_error(ServerError, 503)
            return embed_response([vector_of(1.0) for _ in contents])

    stub = FlakyThenOk()
    embedder = make_embedder(stub)
    vectors = asyncio.run(embedder.embed_documents(["a"]))
    assert len(vectors) == 1
    assert stub.calls == 3


def test_rate_limited_run_backs_off_and_recovers(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    class RateLimited:
        def __init__(self) -> None:
            self.calls = 0

        async def embed_content(self, *, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                raise api_error(ClientError, 429)
            return embed_response([vector_of(1.0) for _ in contents])

    stub = RateLimited()
    embedder = make_embedder(stub)
    assert len(asyncio.run(embedder.embed_documents(["a", "b"]))) == 2
    assert stub.calls == 2


def test_non_retryable_error_fails_on_first_attempt(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    class AlwaysBadRequest:
        def __init__(self) -> None:
            self.calls = 0

        async def embed_content(self, *, model, contents, config):
            self.calls += 1
            raise api_error(ClientError, 400)

    stub = AlwaysBadRequest()
    embedder = make_embedder(stub)
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed_documents(["a"]))
    assert stub.calls == 1


# ---------------------------------------------------------------------
#  Token estimation
# ---------------------------------------------------------------------


def test_token_estimate_is_conservative():
    # Pessimistic by design: it must never under-estimate enough to let an
    # over-limit input through the guard.
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 3500) >= 1000
    assert estimate_tokens(MAX_CHARS_PER_INPUT * "a") >= 8191


# ---------------------------------------------------------------------
#  Free-tier pacing
#
#  Gemini counts each embedded TEXT against the per-minute quota, not each
#  HTTP request. A 64-item call spends 64 units of a 100/min budget, so two
#  back-to-back batches trip the limit - which is exactly how the first
#  full-corpus run failed, on batch 2 of 10.
# ---------------------------------------------------------------------


def test_retry_delay_is_read_from_the_structured_error():
    assert retry_delay_from(Exception("{'retryDelay': '31s'}")) == 31.0


def test_retry_delay_is_read_from_the_prose_message():
    assert retry_delay_from(Exception("Please retry in 31.676545795s.")) == 31.676545795


def test_retry_delay_absent_returns_none():
    assert retry_delay_from(Exception("some other failure")) is None


def test_limiter_allows_a_batch_within_budget():
    limiter = _RateLimiter(100)
    assert limiter.wait_time(64, 0.0) == 0.0


def test_limiter_holds_a_second_batch_that_would_exceed_the_quota():
    """64 + 64 = 128 against a 100/min budget."""
    limiter = _RateLimiter(100)
    limiter.record(64, 0.0)
    assert limiter.wait_time(64, 1.0) > 0


def test_limiter_releases_once_the_window_has_passed():
    limiter = _RateLimiter(100)
    limiter.record(64, 0.0)
    assert limiter.wait_time(64, 61.0) == 0.0


def test_limiter_is_disabled_by_a_non_positive_limit():
    assert _RateLimiter(0).wait_time(10_000, 0.0) == 0.0


def test_run_paces_itself_instead_of_tripping_the_quota(monkeypatch):
    slept: list[float] = []

    def record_sleep(seconds):
        slept.append(seconds)
        return _REAL_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    stub = StubEmbeddings()
    embedder = make_embedder(stub)
    embedder._limiter = _RateLimiter(100)

    # 128 items in two 64-item batches: the second must wait.
    asyncio.run(embedder.embed_documents(["clause"] * 128, batch_size=64))

    assert stub.calls == 2
    assert slept, "second batch should have been paced"
    assert max(slept) > 30


def test_server_retry_delay_is_honoured_over_exponential_backoff(monkeypatch):
    """Backoff alone retried in seconds against a window needing thirty."""
    slept: list[float] = []

    def record_sleep(seconds):
        slept.append(seconds)
        return _REAL_SLEEP(0)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    class QuotaThenOk:
        def __init__(self) -> None:
            self.calls = 0

        async def embed_content(self, *, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                raise api_error(ClientError, 429)
            return embed_response([vector_of(1.0) for _ in contents])

    stub = QuotaThenOk()
    embedder = make_embedder(stub)
    embedder._limiter = _RateLimiter(0)  # isolate backoff from pacing

    # Message carries the provider's requested wait.
    original = api_error(ClientError, 429)
    original.args = ("429 RESOURCE_EXHAUSTED {'retryDelay': '31s'}",)

    class WithDelay(QuotaThenOk):
        async def embed_content(self, *, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                raise original
            return embed_response([vector_of(1.0) for _ in contents])

    stub = WithDelay()
    embedder = make_embedder(stub)
    embedder._limiter = _RateLimiter(0)
    asyncio.run(embedder.embed_documents(["a"]))

    assert slept and 31.0 <= max(slept) < 33.0


# ---------------------------------------------------------------------
#  Daily quota — a clean stop, not a failure
#
#  The free tier allows 1000 embedded items per day. With the trickle
#  strategy this is an expected end-of-run condition, so it must not burn
#  retries or surface as a stack trace.
# ---------------------------------------------------------------------


def test_daily_quota_is_distinguished_from_the_per_minute_limit():
    per_day = "quotaId: 'EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier'"
    per_min = "quotaId: 'EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier'"
    assert is_daily_quota(Exception(per_day)) is True
    assert is_daily_quota(Exception(per_min)) is False


def test_daily_quota_stops_immediately_without_burning_retries(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    class DailyQuota:
        def __init__(self) -> None:
            self.calls = 0

        async def embed_content(self, *, model, contents, config):
            self.calls += 1
            err = api_error(ClientError, 429)
            err.args = (
                "429 RESOURCE_EXHAUSTED quotaId: "
                "'EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier'",
            )
            raise err

    stub = DailyQuota()
    embedder = make_embedder(stub)
    embedder._limiter = _RateLimiter(0)

    with pytest.raises(EmbeddingQuotaExhausted, match="re-run ingestion"):
        asyncio.run(embedder.embed_documents(["a"]))

    # One attempt, not five: the counter resets hours from now.
    assert stub.calls == 1


def test_quota_exhausted_is_an_embedding_error_subclass():
    """Callers that catch EmbeddingError broadly still handle it."""
    assert issubclass(EmbeddingQuotaExhausted, EmbeddingError)
