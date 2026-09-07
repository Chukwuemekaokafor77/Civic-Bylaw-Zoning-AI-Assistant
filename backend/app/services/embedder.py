"""Gemini embedding wrapper (Phase 2, Step 1).

Single entry point for turning text into `VECTOR(1536)` values. Both sides
of retrieval go through here - the ingestion CLI embedding thousands of
bylaw clauses, and the request path embedding one user query - so that
document and query vectors can never be produced by different models or
dimensions. A mismatch there does not raise; it silently returns bad
neighbours, which for this application means confidently-cited wrong law.

Provider note: Section 1 of the spec names OpenAI `text-embedding-3-small`.
This uses Gemini `gemini-embedding-001` instead, at explicit user
direction, because it has a free tier that needs no prepayment. The switch
is deliberately invisible to the rest of the system:

* 1536 dimensions is a supported Matryoshka output size, so `VECTOR(1536)`,
  the HNSW cosine index and the `<=>` operator in `match_bylaw_chunks` are
  all untouched.
* The 8,192-token input ceiling matches the OpenAI model's, so the chunk
  size guard below is unchanged.
* The model is multilingual, which the locked EN/FR scope requires.

`gemini-embedding-001` is chosen over the newer `gemini-embedding-2`
deliberately. Given a list of inputs, `-2` returns ONE aggregated vector
rather than one vector per input, and it does not accept `task_type`.
Aggregation would silently collapse a batch of clauses into a single
meaningless embedding, so the count check in `_embed_batch` is treated as
load-bearing rather than defensive.

Design notes:

* Batched, with an order guard. Unlike OpenAI, Gemini returns no per-item
  index - results are positional only. There is therefore nothing to sort
  by, and a count mismatch is the only available signal that the response
  does not line up with the request. It is treated as fatal: pairing a
  clause with another clause's vector corrupts citations in a way nothing
  short of a semantic eval would catch.

* Task-typed. Documents are embedded as RETRIEVAL_DOCUMENT and queries as
  RETRIEVAL_QUERY, which is what the model expects for asymmetric search.

* Normalised. Truncated Matryoshka outputs from this model are not unit
  length. Cosine distance is magnitude-invariant so ranking would survive,
  but normalising keeps stored vectors consistent and makes the RPC's
  `1 - (embedding <=> query)` a true similarity in [0, 1].

* Fails loudly on oversized input. Silently truncating a clause would drop
  bylaw text from the indexed corpus while still producing a plausible
  vector and a valid-looking citation. An oversized clause is the
  chunker's problem to fix, so this raises instead.
"""

from __future__ import annotations

import asyncio
import collections
import math
import random
import re
import time
from functools import lru_cache

import structlog
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
#  Provider limits
#
#  gemini-embedding-001 accepts 8,192 tokens per input. The batch
#  defaults sit well under the request ceiling: legal clauses carry the
#  Section 4 context prefix and can be long, and a smaller batch fails
#  cheaper when a run hits the free-tier rate limit mid-ingestion.
# ---------------------------------------------------------------------

MAX_TOKENS_PER_INPUT = 8191
MAX_INPUTS_PER_REQUEST = 100

# Sized to divide the free-tier per-minute item quota exactly. At 64 the
# second batch of every minute overruns 100 and stalls for the rest of the
# window, yielding 64 items/min; at 50 two batches fit per window and the
# run sustains the full 100.
DEFAULT_BATCH_SIZE = 50
DEFAULT_BATCH_TOKEN_BUDGET = 18_000

# Task types for asymmetric retrieval.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Token estimation without pulling in a tokenizer. 3.5 chars/token is
# deliberately pessimistic: English prose runs about 4, but French bylaw
# text (Moncton, Fredericton) and citation-dense strings like "6.3(1)(a)"
# tokenize harder. Over-estimating costs a slightly smaller batch;
# under-estimating costs a rejected request.
CHARS_PER_TOKEN_ESTIMATE = 3.5
MAX_CHARS_PER_INPUT = int(MAX_TOKENS_PER_INPUT * CHARS_PER_TOKEN_ESTIMATE)

# Retry policy for transient provider failures.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 65.0

# Free-tier quota exhaustion arrives as HTTP 429.
RATE_LIMIT_STATUS = 429

# The free-tier embedding quota is counted PER TEXT, not per HTTP request:
# a single call carrying 64 inputs spends 64 units of
# `EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier`
# (observed limit 100/min). Batching therefore reduces round trips but buys
# no quota headroom at all, so the pacing below is what actually keeps a
# full-corpus ingestion inside the free tier.
FREE_TIER_ITEMS_PER_MINUTE = 100
RATE_WINDOW_SECONDS = 60.0

# Leaves room for clock skew against the provider's own window.
RATE_WINDOW_SAFETY_SECONDS = 1.0

# Google returns the wait it wants in the error body, in either of these
# shapes. Honouring it beats guessing: blind exponential backoff was
# retrying after a few seconds against a window that needed thirty.
_RETRY_DELAY_PATTERNS = (
    re.compile(r"'retryDelay'\s*:\s*'(\d+(?:\.\d+)?)s'"),
    re.compile(r"retry in (\d+(?:\.\d+)?)s"),
)


def retry_delay_from(exc: Exception) -> float | None:
    """The provider's requested wait, if it stated one."""
    text = str(exc)
    for pattern in _RETRY_DELAY_PATTERNS:
        found = pattern.search(text)
        if found:
            return float(found.group(1))
    return None


class _RateLimiter:
    """Sliding-window limiter over items embedded per minute."""

    def __init__(self, limit: int = FREE_TIER_ITEMS_PER_MINUTE) -> None:
        self.limit = limit
        self._events: collections.deque[tuple[float, int]] = collections.deque()

    def _spent(self, now: float) -> int:
        while self._events and now - self._events[0][0] >= RATE_WINDOW_SECONDS:
            self._events.popleft()
        return sum(count for _, count in self._events)

    def wait_time(self, items: int, now: float) -> float:
        """Seconds to wait before `items` more may be sent."""
        if self.limit <= 0:
            return 0.0
        spent = self._spent(now)
        if spent + items <= self.limit:
            return 0.0
        # Wait for the oldest events to age out of the window.
        needed = spent + items - self.limit
        released = 0
        for timestamp, count in self._events:
            released += count
            if released >= needed:
                age = now - timestamp
                return max(0.0, RATE_WINDOW_SECONDS - age) + RATE_WINDOW_SAFETY_SECONDS
        return RATE_WINDOW_SECONDS + RATE_WINDOW_SAFETY_SECONDS

    def record(self, items: int, now: float) -> None:
        self._events.append((now, items))


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingConfigError(EmbeddingError):
    """Configuration is wrong or incomplete - not retryable."""


class EmbeddingInputTooLarge(EmbeddingError):
    """A single input exceeds the model's per-input token limit."""


class EmbeddingQuotaExhausted(EmbeddingError):
    """The DAILY free-tier quota is spent; nothing will succeed until reset.

    Distinct from the per-minute limit, which pacing and a short wait
    resolve. Retrying this one just burns the remaining attempts against a
    counter that resets hours from now, so the run stops cleanly and says
    what remains instead of failing with a stack trace. Ingestion is
    resumable: the content hash is recorded only for documents that
    finished, so the next run skips them and picks up where this left off.
    """


# The 429 body names which quota was hit. Per-day is terminal for this run.
_DAILY_QUOTA_MARKERS = ("PerDay", "RequestsPerDay")


def is_daily_quota(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def estimate_tokens(text: str) -> int:
    """Approximate token count. Used only for batch sizing and guardrails."""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


def normalize(vector: list[float]) -> list[float]:
    """Scale to unit length, leaving a zero vector alone."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _is_retryable(exc: Exception) -> bool:
    """Transient provider conditions worth another attempt.

    A malformed request or a bad key fails identically five times, so only
    server faults and rate limiting are retried. On the free tier a 429 is
    an expected part of a large ingestion run, not an error.
    """
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        return getattr(exc, "code", None) == RATE_LIMIT_STATUS
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        return code is not None and code >= 500
    return False


class Embedder:
    """Async wrapper over the Gemini embeddings endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        if self._settings.gemini_api_key is None:
            # config.py types this Optional because Phase 1 booted without
            # it. From Phase 2 on it is mandatory, so say so precisely
            # rather than letting a None reach the SDK as an opaque 401.
            raise EmbeddingConfigError(
                "GEMINI_API_KEY is not set. It is required from Phase 2 "
                "onward for embedding generation. A free key needs no "
                "payment method: https://aistudio.google.com/apikey"
            )

        self.model = self._settings.embedding_model
        self.dimensions = self._settings.embedding_dimensions

        self._client = genai.Client(
            api_key=self._settings.gemini_api_key.get_secret_value()
        )
        self._limiter = _RateLimiter(self._settings.embedding_items_per_minute)

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user query.

        Note the deliberate asymmetry with ingestion. The task type differs
        (RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT), and the Section 4
        `[Province: ...] [Municipality: ...]` prefix stored on chunks is
        NOT added here: both RPCs already filter by `municipality_id`, so
        prefixing the query would only spend similarity budget on
        jurisdiction terms every candidate chunk in scope already shares.
        """
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingError("Cannot embed empty query text.")

        vectors = await self._embed([cleaned], task=TASK_QUERY)
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
            task=TASK_DOCUMENT,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
        )

    async def aclose(self) -> None:
        """Present for symmetry; the SDK manages its own transport."""
        return None

    # -----------------------------------------------------------------
    #  Internals
    # -----------------------------------------------------------------

    async def _embed(
        self,
        texts: list[str],
        *,
        task: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_token_budget: int = DEFAULT_BATCH_TOKEN_BUDGET,
    ) -> list[list[float]]:
        if not texts:
            return []

        self._validate_inputs(texts)

        batches = self._make_batches(texts, batch_size, batch_token_budget)
        log.info(
            "embedding_started",
            model=self.model,
            task=task,
            inputs=len(texts),
            batches=len(batches),
        )

        vectors: list[list[float]] = []
        for position, batch in enumerate(batches, start=1):
            vectors.extend(await self._embed_batch(batch, task, position, len(batches)))

        log.info(
            "embedding_complete",
            model=self.model,
            task=task,
            inputs=len(texts),
        )
        return vectors

    def _validate_inputs(self, texts: list[str]) -> None:
        for index, text in enumerate(texts):
            if not text or not text.strip():
                # The API rejects empty strings, but a blank chunk is a
                # parser bug worth surfacing at its source index.
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
        size = max(1, min(batch_size, MAX_INPUTS_PER_REQUEST))

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
        task: str,
        position: int,
        total: int,
    ) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Pace against the per-item quota before spending any of it.
            pause = self._limiter.wait_time(len(batch), time.monotonic())
            if pause > 0:
                log.info(
                    "embedding_paced",
                    batch=f"{position}/{total}",
                    sleep_seconds=round(pause, 1),
                    reason="free-tier per-minute item quota",
                )
                await asyncio.sleep(pause)

            self._limiter.record(len(batch), time.monotonic())
            try:
                response = await self._client.aio.models.embed_content(
                    model=self.model,
                    contents=list(batch),
                    config=types.EmbedContentConfig(
                        task_type=task,
                        # Sent explicitly rather than relying on the model
                        # default (3072), which would not fit VECTOR(1536).
                        output_dimensionality=self.dimensions,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = exc

                if is_daily_quota(exc):
                    raise EmbeddingQuotaExhausted(
                        "Daily free-tier embedding quota exhausted "
                        f"(batch {position}/{total}). The counter resets on "
                        "Google's schedule; re-run ingestion then. Documents "
                        "already committed are recorded and will be skipped."
                    ) from exc

                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    break

                # Prefer the wait the provider actually asked for. Its
                # 429 states a retryDelay (observed: 31s); exponential
                # backoff alone retried far sooner and simply burned the
                # remaining attempts against a window that had not reset.
                stated = retry_delay_from(exc)
                if stated is not None:
                    delay = stated + random.uniform(0, 1.0)
                else:
                    # Full jitter, so a large run does not retry every
                    # batch in lockstep and trip the limit again.
                    ceiling = min(
                        BACKOFF_CAP_SECONDS,
                        BACKOFF_BASE_SECONDS * 2 ** (attempt - 1),
                    )
                    delay = random.uniform(0, ceiling)
                log.warning(
                    "embedding_retry",
                    batch=f"{position}/{total}",
                    attempt=attempt,
                    error=type(exc).__name__,
                    sleep_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)
                continue

            embeddings = list(response.embeddings or [])

            # Gemini returns no per-item index, so results are positional
            # and there is nothing to re-sort by. A count mismatch is the
            # only signal that the response does not correspond to the
            # request - notably if a model that aggregates a list into one
            # vector were ever configured here.
            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"Provider returned {len(embeddings)} embeddings for "
                    f"{len(batch)} inputs; refusing to align them by "
                    "position. Check that the configured model returns one "
                    "embedding per input rather than an aggregate."
                )

            vectors: list[list[float]] = []
            for offset, embedding in enumerate(embeddings):
                values = list(embedding.values or [])
                if len(values) != self.dimensions:
                    raise EmbeddingError(
                        f"Embedding {offset} in batch {position} has "
                        f"{len(values)} dimensions, expected "
                        f"{self.dimensions}. It would be rejected by "
                        "bylaw_chunks.embedding VECTOR(1536)."
                    )
                vectors.append(normalize(values))

            log.debug(
                "embedding_batch_complete",
                batch=f"{position}/{total}",
                inputs=len(batch),
            )
            return vectors

        raise EmbeddingError(
            f"Embedding batch {position}/{total} failed after "
            f"{MAX_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


@lru_cache
def get_embedder() -> Embedder:
    """Cached accessor so one client is shared."""
    return Embedder()
