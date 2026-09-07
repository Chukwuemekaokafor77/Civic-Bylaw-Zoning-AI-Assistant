"""Prompt assembly and Groq streaming (Phase 3, Step 3).

Implements the Section 5 system prompt and guardrails, then streams tokens
from `llama-3.3-70b-versatile`.

Three deliberate departures from the literal spec text, each because the
literal version would produce a less honest answer:

1. The disclaimer is enforced in code, not merely instructed. Rule 6 tells
   the model to append it; models drop trailing instructions, especially
   after a long grounded answer. For the one sentence telling a member of
   the public how far to trust advice about their property, "usually
   present" is not good enough - so it is appended deterministically if
   the model omits it, and the model is still instructed to produce it.

2. The disclaimer states when the date is unknown. Rule 6 interpolates
   `bylaw_last_verified_at`, which is NULL for every municipality until a
   human confirms currency. Rendering "last verified on None", or quietly
   substituting today's date, would assert a human check that never
   happened. The null case gets its own wording instead.

3. The "not found" fallback is returned without calling the model at all
   when retrieval came back empty. Rule 1 prescribes exact wording; asking
   a model to reproduce a fixed string is strictly worse than returning
   it, and it avoids paying for a call whose answer is already known.

The prompt is also parameterised by province rather than hard-coding
"Atlantic Canada", since Phase 0 locked scope to all of Canada.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

import structlog
from groq import AsyncGroq
from groq import APIError as GroqAPIError

from app.config import Settings, get_settings
from app.models.schemas import Chunk

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
#  Fixed strings (Section 5)
# ---------------------------------------------------------------------

FALLBACK_MESSAGE = {
    "en": (
        "I could not find a specific rule covering this in the indexed "
        "zoning bylaws for this municipality. Please contact the local "
        "Municipal Planning & Development department directly."
    ),
    "fr": (
        "Je n'ai pas trouvé de règle précise à ce sujet dans les arrêtés "
        "de zonage indexés pour cette municipalité. Veuillez communiquer "
        "directement avec le service municipal d'urbanisme et "
        "d'aménagement."
    ),
}

DISCLAIMER_VERIFIED = {
    "en": (
        "\n\n*Disclaimer: This AI summary is for informational purposes only "
        "and reflects bylaw text last verified on {verified_on}. Official "
        "verification should always be confirmed with local municipal "
        "planning staff.*"
    ),
    "fr": (
        "\n\n*Avis : Ce résumé généré par IA est fourni à titre informatif "
        "seulement et reflète le texte de l'arrêté vérifié pour la dernière "
        "fois le {verified_on}. Toute vérification officielle doit être "
        "confirmée auprès du personnel municipal d'urbanisme.*"
    ),
}

# Used when bylaw_last_verified_at is NULL. Says plainly that no human has
# confirmed currency, rather than printing a date nobody stands behind.
DISCLAIMER_UNVERIFIED = {
    "en": (
        "\n\n*Disclaimer: This AI summary is for informational purposes only. "
        "The indexed bylaw text has NOT yet been verified against the "
        "municipality's current consolidation, so it may be out of date. "
        "Always confirm with local municipal planning staff before relying "
        "on it.*"
    ),
    "fr": (
        "\n\n*Avis : Ce résumé généré par IA est fourni à titre informatif "
        "seulement. Le texte indexé n'a PAS encore été vérifié par rapport à "
        "la codification administrative en vigueur et pourrait être périmé. "
        "Confirmez toujours auprès du personnel municipal d'urbanisme avant "
        "de vous y fier.*"
    ),
}

# Marker used to detect whether the model already produced the disclaimer.
_DISCLAIMER_MARKERS = ("*Disclaimer:", "*Avis :", "*Avis:")

SYSTEM_PROMPT = """You are the official Civic Zoning & Bylaw Assistant for Canadian municipalities. You are currently answering for {municipality_name}, {province_name}.

Your goal is to provide accurate, accessible answers regarding land use, secondary suites/ADUs, building setbacks, home businesses, and local zoning rules based ONLY on official municipal bylaws provided in the context below.

CONTEXT FROM OFFICIAL BYLAWS:
-----------------------------
{retrieved_chunks}
-----------------------------

STRICT RESPONSE RULES:
1. Rely EXCLUSIVELY on the provided CONTEXT. If the context does not contain sufficient details to answer the user's query, state: "{fallback}"
2. DO NOT make up rules, speculate, or mix bylaws from different municipalities or provinces. Every chunk above is from {municipality_name} only.
3. INLINE CITATIONS ARE MANDATORY: Every requirement, measurement, or rule stated MUST end with an inline citation formatted as: [{municipality_name} - Bylaw Name, Section X.Y]. Copy the CITE AS label shown with each context passage verbatim, including its plain ASCII square brackets [ and ]. Do NOT substitute typographic or full-width brackets such as 【 】, and do not reformat the label.
4. If retrieved chunks conflict with each other (e.g., an amended vs. original clause), surface both and flag the conflict explicitly rather than silently picking one.
5. FORMATTING: Use clear markdown bullet points, bold key terms, and summarize complex legal jargon into plain language.
6. A permitted use and a conditional use are NOT the same thing. A conditional use requires approval and may be refused. Never describe a conditional use as permitted.
7. Answer in {language_name}.
8. MANDATORY DISCLAIMER: Append this disclaimer at the very end of your response, exactly as written:
{disclaimer}"""

PROVINCE_NAMES = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
}

LANGUAGE_NAMES = {"en": "English", "fr": "French"}


@dataclass
class GenerationContext:
    """Everything the prompt needs that is not the question or the chunks."""

    municipality_id: str
    municipality_name: str
    province_code: str
    language: str = "en"
    bylaw_last_verified_at: date | str | None = None

    @property
    def province_name(self) -> str:
        return PROVINCE_NAMES.get(self.province_code, self.province_code)

    @property
    def language_name(self) -> str:
        return LANGUAGE_NAMES.get(self.language, "English")

    def disclaimer(self) -> str:
        """The dated disclaimer, or the honest unverified form."""
        language = self.language if self.language in DISCLAIMER_VERIFIED else "en"
        if not self.bylaw_last_verified_at:
            return DISCLAIMER_UNVERIFIED[language]
        stamp = (
            self.bylaw_last_verified_at.isoformat()
            if isinstance(self.bylaw_last_verified_at, date)
            else str(self.bylaw_last_verified_at)
        )
        return DISCLAIMER_VERIFIED[language].format(verified_on=stamp)

    def fallback(self) -> str:
        language = self.language if self.language in FALLBACK_MESSAGE else "en"
        return FALLBACK_MESSAGE[language]


def format_context(chunks: list[Chunk], context: GenerationContext) -> str:
    """Render retrieved chunks with the exact citation label to copy.

    The label is supplied ready-made rather than left for the model to
    assemble from parts. Section 5 Rule 3 makes the citation format
    mandatory, and a model that has to build "[Municipality - Bylaw,
    Section X]" from three separate fields will eventually get one wrong -
    producing a citation that looks authoritative and points nowhere.
    """
    if not chunks:
        return "(no matching bylaw passages were retrieved)"

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        citation = chunk.citation(context.municipality_name)
        header = f"[Passage {index}] CITE AS: {citation}"
        if chunk.section_title:
            header += f"\nSection title: {chunk.section_title}"
        if chunk.page_number:
            header += f"\nPage: {chunk.page_number}"
        blocks.append(f"{header}\n{chunk.chunk_content}")

    return "\n\n---\n\n".join(blocks)


def build_system_prompt(chunks: list[Chunk], context: GenerationContext) -> str:
    return SYSTEM_PROMPT.format(
        municipality_name=context.municipality_name,
        province_name=context.province_name,
        retrieved_chunks=format_context(chunks, context),
        fallback=context.fallback(),
        language_name=context.language_name,
        disclaimer=context.disclaimer().strip(),
    )


def has_disclaimer(text: str) -> bool:
    return any(marker in text for marker in _DISCLAIMER_MARKERS)


class RagEngineError(RuntimeError):
    """Generation failed."""


class RagEngine:
    """Prompt assembly plus streamed generation via Groq."""

    def __init__(self, settings: Settings | None = None, client: AsyncGroq | None = None) -> None:
        self._settings = settings or get_settings()

        if client is not None:
            self._client = client
        else:
            if self._settings.groq_api_key is None:
                raise RagEngineError(
                    "GROQ_API_KEY is not set. It is required from Phase 3 "
                    "onward for answer generation. A free key needs no "
                    "payment method: https://console.groq.com/keys"
                )
            self._client = AsyncGroq(
                api_key=self._settings.groq_api_key.get_secret_value()
            )

    async def stream(
        self,
        query: str,
        chunks: list[Chunk],
        context: GenerationContext,
    ) -> AsyncIterator[str]:
        """Yield answer text incrementally.

        The caller is responsible for wrapping these in SSE frames; this
        deliberately yields plain text so it can be exercised without a
        web layer.
        """
        if not chunks:
            # Rule 1's wording is fixed, so there is nothing for a model to
            # decide here. Returning it directly also guarantees the
            # fallback is exact, which Phase 5 measures.
            log.info(
                "generation_fallback",
                municipality=context.municipality_id,
                language=context.language,
                reason="no chunks retrieved",
            )
            yield context.fallback()
            yield context.disclaimer()
            return

        system_prompt = build_system_prompt(chunks, context)
        emitted: list[str] = []

        log.info(
            "generation_started",
            municipality=context.municipality_id,
            language=context.language,
            model=self._settings.llm_model,
            chunks=len(chunks),
            sections=[c.section_number for c in chunks],
        )

        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            async for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    emitted.append(delta)
                    yield delta
        except GroqAPIError as exc:
            raise RagEngineError(f"Groq generation failed: {exc}") from exc

        answer = "".join(emitted)

        # Enforced, not merely requested. See the module docstring.
        if not has_disclaimer(answer):
            log.warning(
                "disclaimer_appended_by_backend",
                municipality=context.municipality_id,
                detail="model omitted the Rule 6 disclaimer",
            )
            yield context.disclaimer()

    async def generate(
        self,
        query: str,
        chunks: list[Chunk],
        context: GenerationContext,
    ) -> str:
        """Non-streaming convenience wrapper, used by the Phase 5 eval."""
        parts = [part async for part in self.stream(query, chunks, context)]
        return "".join(parts)
