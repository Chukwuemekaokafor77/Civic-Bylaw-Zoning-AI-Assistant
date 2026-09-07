"""Source fetching and change detection (Phase 2, Step 4).

Answers one question before any embedding is paid for: has this
municipality's bylaw actually changed since the last run? Section 4
requires hashing each source and re-ingesting only what moved, both to
avoid re-billing the embedding API on every run and to give a defensible
trigger for re-verification.

The guards here are not hypothetical - each one comes from a failure
observed while verifying the registry (see the `notes` and
`discarded_urls` in municipalities_config.json):

* Redirects are followed and the resolved URL recorded. Summerside serves
  from a CivicLive CDN, and St. John's publishes a stable alias that 302s
  to a dated CDN filename. The stable alias is what gets cited; the
  resolved URL is what reveals the amendment date.

* A 200 is not proof of a document. Charlottetown's GetFile.ashx returns
  HTTP 200 and a login gateway. Ingesting that would fill the corpus with
  a sign-in page and cite it as zoning law, so the body is checked for a
  PDF signature rather than trusted on status code alone.

* A 404 is a re-ingestion signal, not a transient error. CBRM encodes the
  consolidation date in the URL path and Saint John uses a Laserfiche
  docid tied to one consolidation, so in both cases the old URL dies when
  a new version is published. Retrying is useless; a human has to
  re-discover the link. `SourceGone` says exactly that.

On `bylaw_last_verified_at`: this module deliberately does NOT set it. A
successful fetch proves a file was downloadable, not that it is the
in-force consolidation - the registry makes that distinction explicitly,
and Section 5 Rule 6 prints the date to the public as the date the text
was verified. Machine fetches update `last_fetched_at`; only
`mark_human_verified` touches the human-facing date.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum

import httpx
import structlog
from supabase import AsyncClient, create_async_client

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)

SOURCES_TABLE = "municipality_sources"
MUNICIPALITIES_TABLE = "municipalities"

# Every PDF begins with this signature.
PDF_MAGIC = b"%PDF"

# Enough of the body to identify a login page or an error page served as 200.
SNIFF_BYTES = 1024

FETCH_TIMEOUT_SECONDS = 120.0

# Municipal CDNs are slow and these files reach 15 MB; read in blocks
# rather than holding two copies of the document in memory.
STREAM_CHUNK_BYTES = 64 * 1024


class ChangeReason(str, Enum):
    NEW = "new"                      # never ingested before
    CONTENT_CHANGED = "content_changed"
    UNCHANGED = "unchanged"
    FORCED = "forced"                # operator overrode the check


class SourceError(RuntimeError):
    """Base class for source retrieval failures."""


class SourceGone(SourceError):
    """The URL no longer resolves - the consolidation was replaced.

    Distinct from a transient failure: retrying will not help, because the
    document has moved to a URL only a human can discover.
    """


class NotADocument(SourceError):
    """The response was not the document it claimed to be."""


@dataclass
class FetchResult:
    citation_url: str
    resolved_url: str
    content: bytes
    content_hash: str
    content_type: str | None
    fetched_at: datetime

    @property
    def redirected(self) -> bool:
        return self.resolved_url != self.citation_url

    @property
    def size_bytes(self) -> int:
        return len(self.content)


@dataclass
class ChangeDecision:
    should_ingest: bool
    reason: ChangeReason
    current_hash: str
    previous_hash: str | None = None

    @property
    def summary(self) -> str:
        if self.reason is ChangeReason.UNCHANGED:
            return "unchanged - skipping embedding"
        return f"{self.reason.value} - ingesting"


# ---------------------------------------------------------------------
#  Fetching
# ---------------------------------------------------------------------


async def fetch_source(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    expect: str = "pdf",
) -> FetchResult:
    """Download a bylaw source and hash it.

    Raises `SourceGone` when the URL has been retired, and `NotADocument`
    when the response is not the file type the registry declared.
    """
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": "CivicBylawAssistant/1.0 (+ingestion)"},
    )

    try:
        digest = hashlib.sha256()
        body = bytearray()

        async with client.stream("GET", url) as response:
            if response.status_code in (404, 410):
                raise SourceGone(
                    f"{url} returned {response.status_code}. For sources whose "
                    "URL encodes a consolidation date (CBRM) or a document id "
                    "(Saint John), this means a new consolidation was published "
                    "and the registry entry must be re-discovered by hand - "
                    "retrying will not resolve it."
                )
            response.raise_for_status()

            async for block in response.aiter_bytes(STREAM_CHUNK_BYTES):
                digest.update(block)
                body.extend(block)

            resolved_url = str(response.url)
            content_type = response.headers.get("content-type")

        content = bytes(body)
        if expect == "pdf" and not content.startswith(PDF_MAGIC):
            head = content[:SNIFF_BYTES].decode("utf-8", errors="replace")
            looks_like_login = "login" in head.lower() or "sign in" in head.lower()
            raise NotADocument(
                f"{url} returned {len(content)} bytes that are not a PDF "
                f"(content-type: {content_type}). "
                + (
                    "The body looks like a sign-in page - a 200 status is not "
                    "proof the document was served."
                    if looks_like_login
                    else "Refusing to ingest it as bylaw text."
                )
            )

        result = FetchResult(
            citation_url=url,
            resolved_url=resolved_url,
            content=content,
            content_hash=digest.hexdigest(),
            content_type=content_type,
            fetched_at=datetime.now(timezone.utc),
        )

        log.info(
            "source_fetched",
            url=url,
            resolved_url=resolved_url if result.redirected else None,
            bytes=result.size_bytes,
            content_hash=result.content_hash[:12],
        )
        return result

    finally:
        if owns_client:
            await client.aclose()


# ---------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------


class SourceTracker:
    """Reads and records `municipality_sources` state."""

    def __init__(self, client: AsyncClient, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "SourceTracker":
        settings = settings or get_settings()
        client = await create_async_client(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
        )
        return cls(client, settings)

    async def previous_hash(
        self,
        municipality_id: str,
        language: str,
        bylaw_name: str,
    ) -> str | None:
        response = (
            await self._client.table(SOURCES_TABLE)
            .select("content_hash")
            .eq("municipality_id", municipality_id)
            .eq("language", language)
            .eq("bylaw_name", bylaw_name)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0].get("content_hash") if rows else None

    async def decide(
        self,
        municipality_id: str,
        language: str,
        bylaw_name: str,
        current_hash: str,
        *,
        force: bool = False,
    ) -> ChangeDecision:
        """Whether this document needs re-parsing and re-embedding."""
        previous = await self.previous_hash(municipality_id, language, bylaw_name)

        if force:
            reason = ChangeReason.FORCED
        elif previous is None:
            reason = ChangeReason.NEW
        elif previous != current_hash:
            reason = ChangeReason.CONTENT_CHANGED
        else:
            reason = ChangeReason.UNCHANGED

        decision = ChangeDecision(
            should_ingest=reason is not ChangeReason.UNCHANGED,
            reason=reason,
            current_hash=current_hash,
            previous_hash=previous,
        )
        log.info(
            "change_detection",
            municipality=municipality_id,
            language=language,
            bylaw=bylaw_name,
            decision=decision.reason.value,
            should_ingest=decision.should_ingest,
        )
        return decision

    async def record_fetch(
        self,
        municipality_id: str,
        language: str,
        bylaw_name: str,
        fetch: FetchResult,
        *,
        source_type: str = "pdf",
        document_version: str | None = None,
    ) -> None:
        """Persist what was fetched, keyed on (municipality, language, bylaw).

        Written after a successful ingestion, never before: recording the
        hash first would mark the document current while its chunks were
        still the previous version's, and the next run would then skip it.
        """
        row = {
            "municipality_id": municipality_id,
            "language": language,
            "bylaw_name": bylaw_name,
            "source_url": fetch.citation_url,
            "source_type": source_type,
            "content_hash": fetch.content_hash,
            "last_fetched_at": fetch.fetched_at.isoformat(),
        }
        if document_version:
            row["document_version"] = document_version

        await (
            self._client.table(SOURCES_TABLE)
            .upsert(row, on_conflict="municipality_id,language,bylaw_name")
            .execute()
        )
        log.info(
            "source_recorded",
            municipality=municipality_id,
            language=language,
            bylaw=bylaw_name,
            content_hash=fetch.content_hash[:12],
        )

    async def mark_human_verified(
        self,
        municipality_id: str,
        verified_on: date,
        *,
        language: str | None = None,
        bylaw_name: str | None = None,
    ) -> None:
        """Record that a person confirmed the text is the in-force version.

        Deliberately separate from `record_fetch`. Section 5 Rule 6 prints
        this date to the public as the date the bylaw was verified; a
        successful download only proves a file existed at a URL. Setting it
        automatically would turn "we downloaded something" into "a human
        confirmed this is current law" - which is the claim the disclaimer
        makes on the municipality's behalf.
        """
        stamp = verified_on.isoformat()

        await (
            self._client.table(MUNICIPALITIES_TABLE)
            .update({"bylaw_last_verified_at": stamp})
            .eq("id", municipality_id)
            .execute()
        )

        query = (
            self._client.table(SOURCES_TABLE)
            .update({"last_verified_at": stamp})
            .eq("municipality_id", municipality_id)
        )
        if language:
            query = query.eq("language", language)
        if bylaw_name:
            query = query.eq("bylaw_name", bylaw_name)
        await query.execute()

        log.info(
            "source_marked_verified",
            municipality=municipality_id,
            verified_on=stamp,
        )
