"""Keyword retrieval and hybrid merge (Phase 3, Step 2).

Two jobs:

1. Execute `keyword_search_bylaw_chunks`, after pulling any citation-like
   token out of the question ("what does section 6.3 say" -> "6.3") to
   feed the RPC's `section_hint`.

2. Fuse the keyword and vector result lists into one ranking.

Why this exists at all (Section 2): vector search alone under-performs on
exactly the questions this corpus invites. Asked "what does 8.14(4) say",
an embedding model has no special regard for the string "8.14(4)" - it
retrieves clauses that are semantically about standards, from whichever
zone. The keyword RPC scores an exact section match at 1.00, so the clause
the person actually named comes back first.

The reverse also holds, which is why neither side is dropped: "can I put a
granny flat in my back yard" contains no term in the bylaw at all, because
Fredericton calls it a Garden Suite. Only the vector side finds that.

Fusion uses Reciprocal Rank Fusion. RRF combines by RANK rather than by
score, which matters here because the two scores are not commensurable:
the vector side returns cosine similarity (roughly 0.65-0.75 for good
Fredericton hits, never near 1.0) while the keyword side returns a banded
score that is exactly 1.00 for an exact section match. Averaging or
summing those would let any keyword hit outrank every semantic hit,
regardless of whether it was actually relevant.
"""

from __future__ import annotations

import re

import structlog
from supabase import AsyncClient, create_async_client

from app.config import Settings, get_settings
from app.models.schemas import Chunk

log = structlog.get_logger(__name__)

RPC_NAME = "keyword_search_bylaw_chunks"

# RRF's damping constant. 60 is the value from the original paper and the
# usual default; it keeps any single list from dominating on rank alone.
RRF_K = 60

# Citation-like tokens a user might type. Ordered most specific first so
# "6.3(1)(a)" is preferred over the bare "6.3" inside it.
_SECTION_PATTERNS = (
    # 8.14(4)(a) / 6.3(1)(a) — number, dot, number, bracketed parts
    re.compile(r"\b(\d+\.\d+(?:\([0-9a-zA-Z]+\))+)"),
    # 6.3.2 — dotted multi-level
    re.compile(r"\b(\d+\.\d+\.\d+)\b"),
    # 8.14 — the common case
    re.compile(r"\b(\d+\.\d+)\b"),
    # "section 7" — bare number, but ONLY when the word "section" precedes
    # it. Without that guard every "3 metres" in a question would be read
    # as a citation and drag in an unrelated Part 3 clause at score 1.00.
    re.compile(r"\bsections?\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\barticles?\s+(\d{1,2})\b", re.IGNORECASE),
)


def extract_section_hint(query: str) -> str | None:
    """Pull a section number out of a natural-language question.

    Returns None when the question names no section, which is the common
    case; the RPC then runs its text signals only.
    """
    if not query:
        return None
    for pattern in _SECTION_PATTERNS:
        found = pattern.search(query)
        if found:
            return found.group(1)
    return None


class KeywordSearch:
    """Executor for the trigram / full-text / section-number RPC."""

    def __init__(self, client: AsyncClient, settings: Settings | None = None) -> None:
        self._client = client
        self._settings = settings or get_settings()

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "KeywordSearch":
        settings = settings or get_settings()
        client = await create_async_client(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
        )
        return cls(client, settings)

    async def search(
        self,
        query: str,
        municipality_id: str,
        *,
        language: str | None = None,
        section_hint: str | None = None,
        match_count: int | None = None,
    ) -> list[Chunk]:
        """Keyword search within one municipality and language."""
        hint = section_hint if section_hint is not None else extract_section_hint(query)

        params = {
            "search_text": query,
            "target_municipality": municipality_id,
            # As in the vector path: NULL means "any language" in the RPC,
            # which the bilingual lock does not permit.
            "target_language": language or self._settings.default_language,
            "section_hint": hint,
            "match_count": (
                self._settings.keyword_match_count if match_count is None else match_count
            ),
        }

        response = await self._client.rpc(RPC_NAME, params).execute()
        rows = response.data or []

        chunks = [Chunk.model_validate(row) for row in rows]
        for chunk in chunks:
            chunk.score = chunk.rank

        log.info(
            "keyword_search",
            municipality=municipality_id,
            language=params["target_language"],
            section_hint=hint,
            returned=len(chunks),
            top_rank=round(chunks[0].rank, 4) if chunks and chunks[0].rank else None,
        )
        return chunks


# ---------------------------------------------------------------------
#  Fusion
# ---------------------------------------------------------------------


def _raw_score(chunk: Chunk) -> float:
    """The best signal any retriever gave this chunk, for tie-breaking."""
    return max(chunk.similarity or 0.0, chunk.rank or 0.0)


def reciprocal_rank_fusion(
    result_lists: list[list[Chunk]],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
    limit: int | None = None,
) -> list[Chunk]:
    """Merge ranked lists by rank position rather than by raw score.

    A chunk found by both retrievers accumulates contributions from each,
    so agreement between the two methods is what promotes a result - which
    is the property that makes hybrid retrieval better than either half.

    Deduplication is by chunk id. The same clause reached by both paths
    must collapse into one row, or the prompt would carry it twice and the
    model would read the repetition as two independent provisions saying
    the same thing.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError(
            f"{len(weights)} weights for {len(result_lists)} result lists."
        )

    fused: dict[str, Chunk] = {}
    scores: dict[str, float] = {}

    for results, weight in zip(result_lists, weights):
        for position, chunk in enumerate(results, start=1):
            key = str(chunk.id)
            scores[key] = scores.get(key, 0.0) + weight / (k + position)
            if key not in fused:
                fused[key] = chunk
            else:
                # Keep whichever copy carries the stronger raw signal, so
                # an exact section match is not discarded in favour of the
                # same chunk arriving from the vector side with only a
                # similarity set.
                if _raw_score(chunk) > _raw_score(fused[key]):
                    merged = chunk.model_copy()
                    merged.similarity = chunk.similarity or fused[key].similarity
                    merged.rank = chunk.rank or fused[key].rank
                    fused[key] = merged

    # Tie-break on the strongest raw signal. Both lists put their best hit
    # at rank 1, so a question naming a section produces a dead heat
    # between the exact match (rank 1.00) and an unrelated clause that
    # merely reads as similar (similarity ~0.66). Ordering by raw score
    # inside a tie puts the section the person actually asked about first.
    ranked = sorted(
        fused.values(),
        key=lambda c: (scores[str(c.id)], _raw_score(c)),
        reverse=True,
    )
    for chunk in ranked:
        chunk.score = scores[str(chunk.id)]

    if limit is not None:
        ranked = ranked[:limit]

    log.info(
        "hybrid_merge",
        inputs=[len(r) for r in result_lists],
        unique=len(fused),
        returned=len(ranked),
    )
    return ranked
