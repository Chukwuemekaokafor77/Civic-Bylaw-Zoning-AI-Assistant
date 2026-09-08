#!/usr/bin/env python
"""Universal bylaw ingestion CLI (Phase 2, Step 5).

Drives the whole pipeline for one municipality or the whole registry:

    registry sync -> fetch -> change detection -> parse -> chunk
                  -> embed -> upsert -> record hash

Usage
-----
    python scripts/ingest_bylaws.py --all
    python scripts/ingest_bylaws.py --municipality nb_fredericton
    python scripts/ingest_bylaws.py --municipality nb_fredericton --language en
    python scripts/ingest_bylaws.py --all --dry-run       # no writes, no embedding
    python scripts/ingest_bylaws.py --municipality X --force   # ignore the hash

Ordering matters and is not arbitrary:

* The registry is synced before any chunk is written, because
  `bylaw_chunks.municipality_id` is a foreign key - chunks for an
  unregistered municipality are rejected by the database, not silently
  orphaned.

* The content hash is recorded LAST, only after the chunks are committed.
  Recording it earlier would mark a document current while its chunks
  were still the previous version's, and every later run would then skip
  it - leaving stale bylaw text served as current indefinitely.

* Municipalities the registry marks `is_active: false` are skipped.
  Halifax and Charlottetown are blocked for stated reasons (no single
  municipality-wide bylaw; no citable current source), and ingesting them
  anyway would put confidently-cited wrong law in front of the public.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Allow `python scripts/ingest_bylaws.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.chunker import MunicipalityContext, chunk_clauses  # noqa: E402
from app.services.document_parser import (  # noqa: E402
    SCHEME_HEALTH_MAX_CHARS,
    parse_pdf,
    scheme_for,
    scheme_health,
)
from app.services.embedder import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    Embedder,
    EmbeddingQuotaExhausted,
)
from app.services.source_tracker import (  # noqa: E402
    NotADocument,
    SourceGone,
    SourceTracker,
    fetch_source,
)
from app.services.vector_store import BylawChunkStore  # noqa: E402

log = structlog.get_logger("ingest")

CONFIG_PATH = Path(__file__).resolve().parent / "municipalities_config.json"

MUNICIPALITIES_TABLE = "municipalities"


@dataclass
class SourceOutcome:
    municipality_id: str
    language: str
    bylaw_name: str
    status: str
    clauses: int = 0
    chunks: int = 0
    upserted: int = 0
    deleted: int = 0
    resumed: int = 0
    detail: str | None = None


@dataclass
class RunSummary:
    outcomes: list[SourceOutcome] = field(default_factory=list)

    def add(self, outcome: SourceOutcome) -> None:
        self.outcomes.append(outcome)

    @property
    def failed(self) -> list[SourceOutcome]:
        return [o for o in self.outcomes if o.status in ("error", "source_gone", "not_a_document")]

    @property
    def quota_blocked(self) -> list[SourceOutcome]:
        return [o for o in self.outcomes if o.status == "quota_exhausted"]

    def render(self) -> str:
        width = 92
        lines = ["", "=" * width, "INGESTION SUMMARY", "=" * width]
        lines.append(
            f"{'municipality':<20}{'lang':<6}{'status':<16}"
            f"{'clauses':>8}{'chunks':>8}{'upsert':>8}{'del':>6}"
        )
        lines.append("-" * width)
        for o in self.outcomes:
            lines.append(
                f"{o.municipality_id:<20}{o.language:<6}{o.status:<16}"
                f"{o.clauses:>8}{o.chunks:>8}{o.upserted:>8}{o.deleted:>6}"
            )
            if o.detail:
                lines.append(f"    -> {o.detail}")
        lines.append("=" * width)

        total = sum(o.upserted for o in self.outcomes)
        skipped = sum(1 for o in self.outcomes if o.status == "unchanged")
        lines.append(
            f"{len(self.outcomes)} source(s): {total} chunks written, "
            f"{skipped} unchanged, {len(self.failed)} failed"
        )
        return "\n".join(lines)


def load_registry(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sources_for(entry: dict) -> list[dict]:
    """Per-language source documents, falling back to the top-level fields."""
    sources = entry.get("sources") or []
    if sources:
        return sources
    if entry.get("source_url"):
        return [
            {
                "language": (entry.get("languages") or ["en"])[0],
                "bylaw_name": entry["source_bylaw_name"],
                "source_url": entry["source_url"],
                "source_type": "pdf",
            }
        ]
    return []


def deduplicate_sources(sources: list[dict]) -> list[dict]:
    """Collapse languages that share one physical document.

    Moncton registers EN and FR against the SAME interleaved PDF. Fetching
    and embedding it twice would double the cost and write two chunk sets
    whose text is identical but whose `language` differs, so a French query
    could retrieve the English row and cite it as French law. Splitting that
    document by language is a parser problem (flagged in the registry), not
    something a second download solves.
    """
    seen: dict[str, dict] = {}
    collapsed: list[dict] = []
    for source in sources:
        url = source["source_url"]
        if url in seen:
            seen[url].setdefault("_also_languages", []).append(source["language"])
            continue
        entry = dict(source)
        seen[url] = entry
        collapsed.append(entry)
    return collapsed


async def sync_registry(client, entry: dict) -> None:
    """Upsert one municipality row so chunk foreign keys resolve."""
    row = {
        "id": entry["id"],
        "province_code": entry["province_code"],
        "name": entry["name"],
        "source_bylaw_name": entry["source_bylaw_name"],
        "source_url": entry.get("source_url") or "",
        "languages": entry.get("languages") or ["en"],
        "is_active": bool(entry.get("is_active")),
    }
    # bylaw_last_verified_at is deliberately NOT written here. It is the
    # date the public disclaimer claims the text was verified, and the
    # registry keeps it null until a human confirms currency.
    await client.table(MUNICIPALITIES_TABLE).upsert(row, on_conflict="id").execute()


async def ingest_source(
    entry: dict,
    source: dict,
    *,
    embedder: Embedder,
    store: BylawChunkStore,
    tracker: SourceTracker,
    force: bool,
    dry_run: bool,
) -> SourceOutcome:
    municipality_id = entry["id"]
    language = source["language"]
    bylaw_name = source["bylaw_name"]

    outcome = SourceOutcome(
        municipality_id=municipality_id,
        language=language,
        bylaw_name=bylaw_name,
        status="pending",
    )

    try:
        fetched = await fetch_source(
            source["source_url"], expect=source.get("source_type", "pdf")
        )
    except SourceGone as exc:
        outcome.status = "source_gone"
        outcome.detail = str(exc)[:200]
        return outcome
    except NotADocument as exc:
        outcome.status = "not_a_document"
        outcome.detail = str(exc)[:200]
        return outcome
    except Exception as exc:  # noqa: BLE001 - isolated deliberately
        # One municipality's problem must not end the run. Moncton's site
        # serves an incomplete TLS chain, and before this branch existed
        # that single failure aborted ingestion for the six municipalities
        # queued behind it.
        outcome.status = "error"
        outcome.detail = f"{type(exc).__name__}: {exc}"[:200]
        return outcome

    decision = await tracker.decide(
        municipality_id, language, bylaw_name, fetched.content_hash, force=force
    )
    if not decision.should_ingest:
        outcome.status = "unchanged"
        outcome.detail = "content hash matches the last ingested version"
        return outcome

    # pdfplumber needs a file path, so the fetched bytes are staged.
    with tempfile.TemporaryDirectory() as workdir:
        pdf_path = Path(workdir) / "source.pdf"
        pdf_path.write_bytes(fetched.content)

        # Numbering differs per municipality; the registry says which.
        clauses = parse_pdf(pdf_path, scheme=scheme_for(entry.get("numbering")))
        outcome.clauses = len(clauses)
        health = scheme_health(clauses)

    if not clauses:
        outcome.status = "no_clauses"
        outcome.detail = (
            "parser produced no clauses - the numbering scheme likely does "
            "not match this municipality's document"
        )
        return outcome

    # A scheme that fits only part of a document still produces clauses,
    # each absorbing everything up to the next heading it recognises. That
    # text then carries the wrong section number, so an answer quoting it
    # cites a provision it did not come from. Refuse rather than publish it.
    if not health["healthy"] and not force:
        outcome.status = "unhealthy_parse"
        outcome.detail = (
            f"{health['oversized']} clause(s) over "
            f"{SCHEME_HEALTH_MAX_CHARS:,} characters (largest {health['max']:,}); "
            "the numbering scheme does not fit this document. Re-run with "
            "--force to ingest anyway."
        )
        return outcome

    context = MunicipalityContext(
        municipality_id=municipality_id,
        municipality_name=entry["name"],
        province_code=entry["province_code"],
        bylaw_name=bylaw_name,
        source_url=source["source_url"],
        language=language,
        document_version=source.get("document_version"),
    )
    payloads = chunk_clauses(clauses, context)
    outcome.chunks = len(payloads)

    if dry_run:
        outcome.status = "dry_run"
        outcome.detail = f"{decision.reason.value}; would embed and upsert"
        return outcome

    # Resume: skip chunks a previous run already embedded and committed.
    done = await store.embedded_chunk_keys(municipality_id, bylaw_name, language)
    pending = [p for p in payloads if (p.section_number, p.chunk_index) not in done]
    outcome.resumed = len(payloads) - len(pending)

    # Embed and commit in slices rather than embedding the whole document
    # and writing once. Kept after the move to a local model: a slice that
    # completes is a slice that survives an interrupted run, and it keeps
    # peak memory bounded on a 2 GB model.
    quota_hit: str | None = None
    for start in range(0, len(pending), DEFAULT_BATCH_SIZE):
        slice_ = pending[start : start + DEFAULT_BATCH_SIZE]
        try:
            vectors = await embedder.embed_documents([p.chunk_content for p in slice_])
        except EmbeddingQuotaExhausted as exc:
            quota_hit = str(exc)
            break
        outcome.upserted += await store.upsert_chunks(slice_, vectors)

    if quota_hit:
        # No hash is recorded, so the next run re-parses and resumes from
        # whatever is already committed.
        outcome.status = "quota_exhausted"
        outcome.detail = (
            f"{outcome.upserted} chunk(s) committed this run, "
            f"{len(pending) - outcome.upserted} still to embed. {quota_hit}"
        )
        return outcome

    # Orphan removal and the hash are deferred until the document is whole:
    # deleting on a partial run would strip chunks that are still valid,
    # and recording the hash would mark an incomplete document as current.
    outcome.deleted = await store.delete_orphaned_chunks(
        municipality_id, bylaw_name, language, payloads
    )

    # Recorded only now that the chunks are committed.
    await tracker.record_fetch(
        municipality_id,
        language,
        bylaw_name,
        fetched,
        source_type=source.get("source_type", "pdf"),
        document_version=source.get("document_version"),
    )

    outcome.status = decision.reason.value
    if fetched.redirected:
        outcome.detail = f"resolved to {fetched.resolved_url}"
    return outcome


async def run(args: argparse.Namespace) -> int:
    registry = load_registry()
    entries = registry["municipalities"]

    if args.municipality:
        entries = [e for e in entries if e["id"] == args.municipality]
        if not entries:
            print(f"No municipality with id {args.municipality!r} in the registry.")
            return 2
    else:
        entries = [e for e in entries if e.get("is_active")]

    settings = get_settings()
    store = await BylawChunkStore.create(settings)
    tracker = await SourceTracker.create(settings)
    embedder = None if args.dry_run else Embedder(settings)

    summary = RunSummary()

    for entry in entries:
        if not entry.get("is_active") and not args.municipality:
            continue

        if not entry.get("is_active"):
            print(
                f"\n{entry['id']}: registry marks this inactive "
                f"({entry.get('status')}). Ingesting it anyway would publish "
                "citations from a source the registry says is not usable."
            )
            if not args.include_inactive:
                summary.add(
                    SourceOutcome(entry["id"], "-", "-", "skipped_inactive",
                                  detail=entry.get("status"))
                )
                continue

        await sync_registry(store._client, entry)

        sources = deduplicate_sources(sources_for(entry))
        if not sources:
            summary.add(
                SourceOutcome(entry["id"], "-", "-", "no_source",
                              detail="registry has no usable source_url")
            )
            continue

        for source in sources:
            if args.language and source["language"] != args.language:
                continue

            print(f"\n>>> {entry['id']} [{source['language']}] {source['bylaw_name']}")
            try:
                outcome = await ingest_source(
                    entry,
                    source,
                    embedder=embedder,
                    store=store,
                    tracker=tracker,
                    force=args.force,
                    dry_run=args.dry_run,
                )
            except Exception as exc:  # noqa: BLE001 - isolated deliberately
                # Second layer: a failure in parsing, chunking or upsert
                # is also contained to its own source.
                outcome = SourceOutcome(
                    entry["id"],
                    source["language"],
                    source["bylaw_name"],
                    "error",
                    detail=f"{type(exc).__name__}: {exc}"[:200],
                )
            if outcome.resumed:
                outcome.detail = (
                    (outcome.detail + "; ") if outcome.detail else ""
                ) + f"{outcome.resumed} chunk(s) already embedded by an earlier run"
            if source.get("_also_languages"):
                shared = ", ".join(source["_also_languages"])
                outcome.detail = (
                    (outcome.detail + "; ") if outcome.detail else ""
                ) + f"same document also registered for: {shared} (not re-ingested)"
            summary.add(outcome)
            print(f"    {outcome.status}: {outcome.chunks} chunks")

    print(summary.render())
    return 1 if summary.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest municipal zoning bylaws.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every active municipality")
    target.add_argument("--municipality", help="a single municipality id")

    parser.add_argument("--language", help="restrict to one language code")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-ingest even when the content hash is unchanged",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and chunk only - no embedding calls and no writes",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="ingest a municipality the registry marks inactive (not advised)",
    )
    args = parser.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
