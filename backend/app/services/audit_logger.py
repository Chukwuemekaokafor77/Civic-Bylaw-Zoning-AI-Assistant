"""Query audit trail (Phase 3, Step 4).

Writes one `query_log` row per answered question: the question, the chunk
ids that were retrieved, the response, and whether the "not found"
fallback fired. Section 7 wants this for three things - seeing which
municipalities and topics are under-covered, knowing how often the
assistant declines to answer, and having a record if a citation is ever
disputed.

Two properties this module is built around:

* Logging never breaks the answer. A failed audit write is logged and
  swallowed. The alternative - a Supabase hiccup turning a correct,
  already-streamed answer into a 500 - trades a real user-facing failure
  for a record-keeping one. The write happens after the response is
  complete, so there is nothing left to salvage by raising.

* The session id is not stored. `ChatRequest.session_id` exists for
  Phase 5 rate limiting and its schema says plainly it is "not a user
  identifier and not stored in query_log". Keeping it here would turn an
  anonymous coverage log into a per-person history of what people asked
  about their own properties, which is a different thing to hold and a
  different thing to be asked for. The table has no column for it, and
  this module does not add one.

`query_log` is backend-only: RLS is enabled with no permissive policy and
anon/authenticated are revoked, so these rows are never reachable from a
browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import structlog
from supabase import AsyncClient, create_async_client

from app.config import Settings, get_settings
from app.models.schemas import Chunk

log = structlog.get_logger(__name__)

TABLE = "query_log"

# Responses are stored for dispute resolution, not analytics, so a runaway
# generation should not write an unbounded row.
MAX_RESPONSE_CHARS = 20_000


@dataclass
class QueryRecord:
    """One answered question, ready to persist."""

    municipality_id: str
    user_query: str
    retrieved_chunk_ids: list[UUID] = field(default_factory=list)
    response_text: str | None = None
    was_fallback: bool = False
    language: str = "en"

    @classmethod
    def from_answer(
        cls,
        *,
        municipality_id: str,
        user_query: str,
        chunks: list[Chunk],
        response_text: str,
        language: str = "en",
        was_fallback: bool | None = None,
    ) -> "QueryRecord":
        """Build a record from what the request actually produced.

        `was_fallback` defaults to "nothing was retrieved". That is the
        condition Section 7 wants counted - the assistant had no grounds to
        answer - and it is derived here rather than passed in so a caller
        cannot forget to set it and quietly under-report coverage gaps.
        """
        return cls(
            municipality_id=municipality_id,
            user_query=user_query,
            retrieved_chunk_ids=[chunk.id for chunk in chunks],
            response_text=response_text,
            was_fallback=not chunks if was_fallback is None else was_fallback,
            language=language,
        )

    def as_row(self) -> dict:
        response = self.response_text
        if response and len(response) > MAX_RESPONSE_CHARS:
            response = response[:MAX_RESPONSE_CHARS] + "... [truncated]"

        return {
            "municipality_id": self.municipality_id,
            "user_query": self.user_query,
            # PostgREST serialises a UUID[] from a list of strings.
            "retrieved_chunk_ids": [str(cid) for cid in self.retrieved_chunk_ids],
            "response_text": response,
            "was_fallback": self.was_fallback,
            "language": self.language,
        }


class AuditLogger:
    """Appends to `query_log`."""

    def __init__(self, client: AsyncClient, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "AuditLogger":
        settings = settings or get_settings()
        client = await create_async_client(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
        )
        return cls(client, settings)

    async def record(self, record: QueryRecord) -> bool:
        """Persist one answered question. Returns whether the write landed.

        Never raises: the answer has already reached the user by the time
        this runs, so failing the request now would convert a bookkeeping
        problem into a user-visible one.
        """
        try:
            await self._client.table(TABLE).insert(record.as_row()).execute()
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            log.error(
                "audit_write_failed",
                municipality=record.municipality_id,
                was_fallback=record.was_fallback,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

        log.info(
            "query_logged",
            municipality=record.municipality_id,
            language=record.language,
            chunks=len(record.retrieved_chunk_ids),
            was_fallback=record.was_fallback,
            response_chars=len(record.response_text or ""),
        )
        return True

    async def fallback_rate(self, municipality_id: str) -> tuple[int, int]:
        """(fallbacks, total) for one municipality - the Phase 6 dashboard.

        A high ratio is the signal Section 7 asks for: it means questions
        are arriving that the indexed corpus cannot answer, which is a
        coverage gap rather than a user error.
        """
        total = (
            await self._client.table(TABLE)
            .select("id", count="exact")
            .eq("municipality_id", municipality_id)
            .limit(1)
            .execute()
        )
        fallbacks = (
            await self._client.table(TABLE)
            .select("id", count="exact")
            .eq("municipality_id", municipality_id)
            .eq("was_fallback", True)
            .limit(1)
            .execute()
        )
        return (fallbacks.count or 0, total.count or 0)
