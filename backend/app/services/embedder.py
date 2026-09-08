"""Voyage embedding wrapper (Phase 2, Step 1).

Single entry point for turning text into `VECTOR(1024)` values. Both sides
of retrieval go through here - the ingestion CLI embedding thousands of
bylaw clauses, and the request path embedding one user query - so that
document and query vectors can never be produced by different models or
dimensions. A mismatch there does not raise; it silently returns bad
neighbours, which for this application means confidently-cited wrong law.

Provider note. Section 1 names OpenAI `text-embedding-3-small`. This has
since run on Gemini and on a local model, and now runs on Voyage, each
change at explicit user direction. The reasons, in order:

* OpenAI needs a $5 prepayment, which was not wanted.
* Gemini's free tier allows 1,000 embeddings per DAY. One bilingual
  municipality is ~1,340 chunks, so the nine-municipality pilot was a
  week of trickling and the national corpus was unreachable.
* A local model (bge-m3) removed the quota but cost 0.6s of CPU per query
  on every user request, and separated a right answer from a wrong one by
  only 0.049 cosine.

Voyage resolves all three: a 200M-token free grant with no daily cap,
0.28s query latency, and a 0.081 margin on the same comparison. It also
emits 1024 dimensions natively, which is what `bylaw_chunks.embedding` was
migrated to in 003 - so this provider change needs no schema change.

Design notes:

* Order-preserving. The API returns an `index` per item and does not
  guarantee response order; results are sorted on it explicitly. Trusting
  arrival order would attach one clause's vector to another clause's row -
  corruption that nothing short of a semantic eval would catch.

* Task-typed. Documents are embedded as `document` and queries as `query`,
  the asymmetric mode the model expects.

* Fails loudly on oversized input. Silently truncating a clause would drop
  bylaw text from the corpus while still producing a plausible vector and
  a valid-looking citation. An oversized clause is the chunker's problem.
"""

from __future__ import annotations

import asyncio
import collections
import random
import time
from functools import lru_cache

import httpx
import structlog

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)

API_URL = "https://api.voyageai.com/v1/embeddings"

# voyage-4 series accepts 32,000 tokens per input, far above any chunk the
# Section 4 chunker produces. The guard below is kept well under it.
MAX_TOKENS_PER_INPUT = 32_000

# Inputs per request. The API accepts up to 1,000, but a batch also has
# to fit inside the account's per-minute token allowance or it can never
# succeed: a 96-chunk batch is ~19K tokens against a 10K/min free-tier
# ceiling, so every request 429s no matter how long the client waits.
# _batch_token_budget() derives the real ceiling from configuration.
DEFAULT_BATCH_SIZE = 96
DEFAULT_BATCH_TOKEN_BUDGET = 100_000

# Leave headroom against the provider's own accounting: the estimate here
# is approximate, and landing exactly on the limit trips it.
TOKEN_BUDGET_SAFETY = 0.85

RATE_WINDOW_SECONDS = 60.0

# Asymmetric retrieval modes.
TASK_DOCUMENT = "document"
TASK_QUERY = "query"

# Token estimation without a tokenizer. 3.5 chars/token is deliberately
# pessimistic: English prose runs about 4, but French bylaw text and
# citation-dense strings like "6.3(1)(a)" tokenize harder.
CHARS_PER_TOKEN_ESTIMATE = 3.5
MAX_CHARS_PER_INPUT = int(MAX_TOKENS_PER_INPUT * CHARS_PER_TOKEN_ESTIMATE)

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 120.0

RATE_LIMIT_STATUS = 429


class _RateLimiter:
    """Sliding window over both requests and tokens per minute.

    Voyage limits accounts without a payment method to 3 RPM and 10K TPM.
    Both have to be respected: pacing on requests alone still trips the
    token ceiling, and pacing on tokens alone still trips the request
    ceiling on a run of small batches.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._events: collections.deque[tuple[float, int]] = collections.deque()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= RATE_WINDOW_SECONDS:
            self._events.popleft()

    def wait_time(self, tokens: int, now: float) -> float:
        """Seconds to wait before a request of `tokens` may be sent."""
        if self.requests_per_minute <= 0 and self.tokens_per_minute <= 0:
            return 0.0

        self._prune(now)
        requests = len(self._events)
        spent = sum(count for _, count in self._events)

        over_requests = (
            self.requests_per_minute > 0 and requests + 1 > self.requests_per_minute
        )
        over_tokens = (
            self.tokens_per_minute > 0 and spent + tokens > self.tokens_per_minute
        )
        if not (over_requests or over_tokens):
            return 0.0

        # Wait for the oldest event to age out of the window; the caller
        # re-checks afterwards, so one step at a time is enough.
        oldest = self._events[0][0] if self._events else now
        return max(0.0, RATE_WINDOW_SECONDS - (now - oldest)) + 0.5

    def record(self, tokens: int, now: float) -> None:
        self._events.append((now, tokens))


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingConfigError(EmbeddingError):
    """Configuration is wrong or incomplete - not retryable."""


class EmbeddingInputTooLarge(EmbeddingError):
    """A single input exceeds the model's per-input token limit."""


class EmbeddingQuotaExhausted(EmbeddingError):
    """The account's token grant is spent.

    Distinct from rate limiting, which a short wait resolves. The
    ingestion CLI stops cleanly on this and reports what remains, rather
    than burning retries against a balance that will not refill on its own.
    """


def estimate_tokens(text: str) -> int:
    """Approximate token count. Used only for batch sizing and guardrails."""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


def _is_quota_exhausted(status: int, body: str) -> bool:
    lowered = body.lower()
    return status in (402, 403) or "quota" in lowered or "insufficient" in lowered


class Embedder:
    """Async wrapper over the Voyage embeddings endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        if self._settings.voyage_api_key is None:
            # config.py types this Optional because Phase 1 booted without
            # it. From Phase 2 on it is mandatory, so say so precisely
            # rather than letting a None reach the API as an opaque 401.
            raise EmbeddingConfigError(
                "VOYAGE_API_KEY is not set. It is required from Phase 2 "
                "onward for embedding generation. A free key with a 200M "
                "token grant is available at https://www.voyageai.com/"
            )

        self.model = self._settings.embedding_model
        self.dimensions = self._settings.embedding_dimensions
        self._limiter = _RateLimiter(
            self._settings.embedding_requests_per_minute,
            self._settings.embedding_tokens_per_minute,
        )
        self._client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={
                "Authorization": (
                    f"Bearer {self._settings.voyage_api_key.get_secret_value()}"
                ),
                "Content-Type": "application/json",
            },
        )

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user query.

        Note the deliberate asymmetry with ingestion. The input type
        differs, and the Section 4 `[Province: ...] [Municipality: ...]`
        prefix stored on chunks is NOT added here: both RPCs already
        filter by `municipality_id`, so prefixing the query would only
        spend similarity budget on jurisdiction terms every candidate
        chunk in scope already shares.
        """
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed empty query text.")

        vectors = await self._embed([cleaned], TASK_QUERY)
        return vectors[0]

    async def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_token_budget: int = DEFAULT_BATCH_TOKEN_BUDGET,
    ) -> list[list[float]]:
        """Embed many chunks, returning vectors in the same order as `texts`."""
        return await self._embed(
            texts,
            TASK_DOCUMENT,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -----------------------------------------------------------------
    #  Internals
    # -----------------------------------------------------------------

    async def _embed(
        self,
        texts: list[str],
        input_type: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_token_budget: int = DEFAULT_BATCH_TOKEN_BUDGET,
    ) -> list[list[float]]:
        if not texts:
            return []

        self._validate_inputs(texts)
        batches = self._make_batches(
            texts, batch_size, self._batch_token_budget(batch_token_budget)
        )

        log.info(
            "embedding_started",
            model=self.model,
            input_type=input_type,
            inputs=len(texts),
            batches=len(batches),
        )

        vectors: list[list[float]] = []
        total_tokens = 0
        for position, batch in enumerate(batches, start=1):
            batch_vectors, used = await self._embed_batch(
                batch, input_type, position, len(batches)
            )
            vectors.extend(batch_vectors)
            total_tokens += used

        log.info(
            "embedding_complete",
            model=self.model,
            inputs=len(texts),
            tokens=total_tokens,
        )
        return vectors

    def _batch_token_budget(self, requested: int) -> int:
        """Cap a batch at what the per-minute allowance can actually pass.

        Without this a batch larger than the whole minute's budget can
        never succeed - it 429s on every attempt and exhausts the retries,
        which is exactly how the first full ingestion run failed.
        """
        ceiling = self._settings.embedding_tokens_per_minute
        if ceiling <= 0:
            return requested
        return max(1, min(requested, int(ceiling * TOKEN_BUDGET_SAFETY)))

    def _validate_inputs(self, texts: list[str]) -> None:
        for index, text in enumerate(texts):
            if not text or not text.strip():
                # A blank chunk is a parser bug worth surfacing at its
                # source index rather than embedding as an empty vector.
                raise EmbeddingError(
                    f"Input at index {index} is empty; a chunk reached the "
                    "embedder with no text. Check clause extraction."
                )
            if len(text) > MAX_CHARS_PER_INPUT:
                raise EmbeddingInputTooLarge(
                    f"Input at index {index} is {len(text)} characters "
                    f"(~{estimate_tokens(text)} tokens), over the "
                    f"{MAX_TOKENS_PER_INPUT}-token limit for {self.model}. "
                    "Split it at a clause boundary in the chunker rather "
                    "than truncating here - truncation would drop bylaw "
                    "text while still yielding a citable-looking chunk."
                )

    def _make_batches(
        self,
        texts: list[str],
        batch_size: int,
        batch_token_budget: int,
    ) -> list[list[str]]:
        """Group inputs under both the per-request count and token ceilings."""
        size = max(1, batch_size)

        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for text in texts:
            tokens = estimate_tokens(text)
            over_budget = bool(current) and current_tokens + tokens > batch_token_budget
            if over_budget or len(current) >= size:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens

        if current:
            batches.append(current)
        return batches

    async def _embed_batch(
        self,
        batch: list[str],
        input_type: str,
        position: int,
        total: int,
    ) -> tuple[list[list[float]], int]:
        tokens = sum(estimate_tokens(text) for text in batch)
        payload = {
            "input": batch,
            "model": self.model,
            "input_type": input_type,
            # Sent explicitly rather than relying on the model default, so
            # a future default change cannot quietly produce vectors that
            # no longer fit VECTOR(1024).
            "output_dimension": self.dimensions,
        }

        last_error: str | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Pace before spending the allowance, not after failing on it.
            pause = self._limiter.wait_time(tokens, time.monotonic())
            if pause > 0:
                log.info(
                    "embedding_paced",
                    batch=f"{position}/{total}",
                    sleep_seconds=round(pause, 1),
                    reason="provider per-minute limit",
                )
                await asyncio.sleep(pause)

            self._limiter.record(tokens, time.monotonic())
            try:
                response = await self._client.post(API_URL, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == MAX_ATTEMPTS:
                    break
                await self._backoff(attempt, position, total, last_error)
                continue

            if response.status_code == 200:
                return self._parse(response.json(), batch, position)

            body = response.text[:400]

            # A spent grant will not refill on its own; retrying only
            # delays telling the operator what actually happened.
            if _is_quota_exhausted(response.status_code, body):
                raise EmbeddingQuotaExhausted(
                    f"Voyage rejected the request as out of quota "
                    f"(HTTP {response.status_code}): {body}"
                )

            retryable = (
                response.status_code == RATE_LIMIT_STATUS
                or response.status_code >= 500
            )
            last_error = f"HTTP {response.status_code}: {body}"
            if not retryable or attempt == MAX_ATTEMPTS:
                break
            await self._backoff(attempt, position, total, last_error)

        raise EmbeddingError(
            f"Embedding batch {position}/{total} failed after "
            f"{MAX_ATTEMPTS} attempts: {last_error}"
        )

    async def _backoff(
        self, attempt: int, position: int, total: int, error: str
    ) -> None:
        # Full jitter, so a large ingestion run that trips a rate limit
        # does not retry every batch in lockstep and trip it again.
        ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
        delay = random.uniform(0, ceiling)
        log.warning(
            "embedding_retry",
            batch=f"{position}/{total}",
            attempt=attempt,
            error=error[:120],
            sleep_seconds=round(delay, 2),
        )
        await asyncio.sleep(delay)

    def _parse(
        self, body: dict, batch: list[str], position: int
    ) -> tuple[list[list[float]], int]:
        # Sorted by the index the API assigns; see the module docstring.
        items = sorted(body.get("data") or [], key=lambda item: item["index"])

        if len(items) != len(batch):
            raise EmbeddingError(
                f"Provider returned {len(items)} embeddings for "
                f"{len(batch)} inputs; refusing to align them by position."
            )

        vectors: list[list[float]] = []
        for offset, item in enumerate(items):
            vector = item.get("embedding") or []
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    f"Embedding {offset} in batch {position} has "
                    f"{len(vector)} dimensions, expected {self.dimensions}. "
                    f"It would be rejected by bylaw_chunks.embedding "
                    f"VECTOR({self.dimensions})."
                )
            vectors.append([float(value) for value in vector])

        used = int((body.get("usage") or {}).get("total_tokens") or 0)
        return vectors, used


@lru_cache
def get_embedder() -> Embedder:
    """Cached accessor so one client (and connection pool) is shared."""
    return Embedder()
