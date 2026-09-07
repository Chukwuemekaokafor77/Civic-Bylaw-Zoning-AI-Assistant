"""Clause-boundary chunking (Phase 2, Step 3).

Turns parsed `Clause` records into rows ready for `bylaw_chunks`, applying
the Section 4 contextual payload and the Section 4 chunking rule: split at
clause and sub-clause boundaries, never at a fixed token count.

Why boundaries rather than a sliding window: a chunk is what the assistant
cites. Cutting "no building shall be located within 3 metres of a rear lot
line" in half yields a fragment that still carries a section number and
still reads like a rule, so a wrong answer arrives correctly formatted and
fully cited. Splitting only where the bylaw itself splits means every
chunk is a complete statement of some rule.

Every chunk repeats its section heading and its parent heading. A
retrieved fragment of "8.14(4) Standards" is meaningless without "8.14
RURAL RESIDENTIAL - CHATEAU HEIGHTS ZONE" attached - "35 % of the lot
area" is a different rule in every zone, and the zone name is the only
thing that distinguishes them.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Iterable

import structlog

from app.services.document_parser import Clause

log = structlog.get_logger(__name__)


# A clause longer than this is split at its sub-clause boundaries. This is
# a retrieval-quality ceiling, not a provider limit: the embedding model
# accepts roughly 28,000 characters, but a 12,000-character chunk dilutes
# its own embedding and forces the reader through pages of irrelevant text
# to reach the sentence that answered the question.
MAX_CHUNK_CHARS = 2400

# Never emit a chunk this small on its own; fold it into its neighbour.
MIN_CHUNK_CHARS = 120

# Sub-clause markers, in the nesting order Z-5 uses:
#   (a) -> (i) -> (A) -> (I), plus the (1) numbering of definitions.
SUBCLAUSE_MARKER = re.compile(
    r"^\((?:[a-z]{1,3}|[A-Z]{1,3}|\d{1,3})\)\s",
)

# Last-resort split point when a single sub-clause still exceeds the
# ceiling. Sentence-final punctuation followed by a capital or a marker.
SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class ChunkPayload:
    """One row destined for `bylaw_chunks`."""

    municipality_id: str
    province_code: str
    bylaw_name: str
    section_number: str
    section_title: str | None
    chunk_content: str
    language: str
    page_number: int | None
    source_document_version: str | None
    metadata: dict
    chunk_index: int

    def as_row(self) -> dict:
        """Column-for-column dict for the Supabase upsert."""
        return {
            "municipality_id": self.municipality_id,
            "province_code": self.province_code,
            "bylaw_name": self.bylaw_name,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "chunk_content": self.chunk_content,
            "language": self.language,
            "page_number": self.page_number,
            "source_document_version": self.source_document_version,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
        }


@dataclass
class MunicipalityContext:
    """Registry facts stamped onto every chunk of one document."""

    municipality_id: str
    municipality_name: str
    province_code: str
    bylaw_name: str
    source_url: str
    language: str = "en"
    document_version: str | None = None
    extra_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
#  Splitting
# ---------------------------------------------------------------------


def _split_on_subclauses(text: str) -> list[str]:
    """Group lines into blocks that each begin at a sub-clause marker.

    Continuation lines - the wrapped remainder of a sub-clause - stay with
    the marker that opened them, which is what keeps a split from landing
    mid-sentence.
    """
    blocks: list[list[str]] = []
    for line in text.split("\n"):
        if SUBCLAUSE_MARKER.match(line.strip()) or not blocks:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return ["\n".join(block).strip() for block in blocks if "\n".join(block).strip()]


def _split_long_block(block: str, limit: int) -> list[str]:
    """Break a single oversized sub-clause at sentence ends."""
    if len(block) <= limit:
        return [block]

    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(block):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    # A single sentence longer than the limit is left intact rather than
    # cut mid-clause; the embedder's own guard is the backstop, and its
    # ceiling is an order of magnitude above this one.
    return pieces


def split_clause_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split one clause body into boundary-aligned pieces under `limit`."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    current = ""

    for block in _split_on_subclauses(text):
        for part in _split_long_block(block, limit):
            candidate = f"{current}\n{part}".strip() if current else part
            if current and len(candidate) > limit:
                pieces.append(current)
                current = part
            else:
                current = candidate

    if current:
        # Fold a stub tail into the previous piece rather than emitting a
        # chunk too short to stand as its own answer.
        if pieces and len(current) < MIN_CHUNK_CHARS:
            pieces[-1] = f"{pieces[-1]}\n{current}"
        else:
            pieces.append(current)

    return pieces


# ---------------------------------------------------------------------
#  Contextual payload (Section 4)
# ---------------------------------------------------------------------


def build_chunk_content(
    clause: Clause,
    context: MunicipalityContext,
    body: str,
    *,
    part: int = 1,
    of: int = 1,
) -> str:
    """Render the Section 4 payload for one chunk.

    The spec's prefix is extended with the parent heading, because Section
    4 also requires "the section heading in every chunk" and for a zone
    standard the parent heading is the zone itself.
    """
    header = (
        f"[Province: {context.province_code}] "
        f"[Municipality: {context.municipality_name}] "
        f"[Bylaw: {context.bylaw_name}]"
    )

    if clause.part_number and clause.part_title:
        header += f" [Part {clause.part_number}: {clause.part_title}]"
    if clause.parent_number and clause.parent_title:
        header += f" [{clause.parent_number} {clause.parent_title}]"

    title = f" {clause.section_title}" if clause.section_title else ""
    marker = f" (part {part} of {of})" if of > 1 else ""

    return f"{header}\nSection {clause.section_number}{title}{marker}:\n{body}"


def _metadata(clause: Clause, context: MunicipalityContext, part: int, of: int) -> dict:
    data = {
        "province": context.province_code,
        "municipality": context.municipality_id,
        "bylaw": context.bylaw_name,
        "section": clause.section_number,
        "section_title": clause.section_title,
        "page": clause.page_number,
        "url": context.source_url,
        "document_version": context.document_version,
        "language": context.language,
    }
    # Only present when the source supplies them; a null in the payload is
    # noise, and these are read back by the citation UI.
    if clause.page_label:
        data["page_label"] = clause.page_label
    if clause.part_number:
        data["part"] = clause.part_number
        data["part_title"] = clause.part_title
    if clause.parent_number:
        data["parent_section"] = clause.parent_number
        data["parent_title"] = clause.parent_title
    if clause.amendments:
        data["amendments"] = clause.amendments
    if of > 1:
        data["chunk_part"] = part
        data["chunk_parts"] = of
    data.update(context.extra_metadata)
    return data


# ---------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------


def chunk_clauses(
    clauses: Iterable[Clause],
    context: MunicipalityContext,
    *,
    limit: int = MAX_CHUNK_CHARS,
) -> list[ChunkPayload]:
    """Convert parsed clauses into `bylaw_chunks` rows."""
    payloads: list[ChunkPayload] = []
    split_count = 0

    # chunk_index must be unique per section number, not per clause. A
    # bylaw can legitimately carry the same number more than once -
    # Z-5 has three separate sign-permission tables all numbered 6.4, one
    # per zone group - and each restarting at 0 collides on the unique
    # natural key, so the upsert would keep one table and drop the rest.
    next_index: dict[str, int] = collections.defaultdict(int)

    for clause in clauses:
        bodies = split_clause_text(clause.text, limit)
        if not bodies:
            continue
        if len(bodies) > 1:
            split_count += 1

        for index, body in enumerate(bodies):
            chunk_index = next_index[clause.section_number]
            next_index[clause.section_number] += 1
            payloads.append(
                ChunkPayload(
                    municipality_id=context.municipality_id,
                    province_code=context.province_code,
                    bylaw_name=context.bylaw_name,
                    section_number=clause.section_number,
                    section_title=clause.section_title or None,
                    chunk_content=build_chunk_content(
                        clause,
                        context,
                        body,
                        part=index + 1,
                        of=len(bodies),
                    ),
                    language=context.language,
                    page_number=clause.page_number,
                    source_document_version=context.document_version,
                    metadata=_metadata(clause, context, index + 1, len(bodies)),
                    # Completes the natural key
                    # (municipality, bylaw, section, language, chunk_index)
                    # so re-ingestion updates rows instead of duplicating.
                    chunk_index=chunk_index,
                )
            )

    log.info(
        "clauses_chunked",
        municipality=context.municipality_id,
        language=context.language,
        # Distinct section numbers, not clause count: a bylaw can number
        # two provisions alike (Z-5 has three tables at 6.4), so this is
        # deliberately labelled for what it measures.
        sections=len({p.section_number for p in payloads}),
        chunks=len(payloads),
        split_clauses=split_count,
    )
    return payloads
