"""Unit tests for app.services.rag_engine (Phase 3, Step 3).

Hermetic: the Groq client is a stub, so no key and no network are needed.
These assert the Section 5 guardrails hold as properties of the code
rather than as instructions a model may or may not follow.
"""

from __future__ import annotations

import asyncio
import types
from datetime import date
from uuid import uuid4

import pytest

from app.config import Settings
from app.models.schemas import Chunk
from app.services.rag_engine import (
    DISCLAIMER_UNVERIFIED,
    FALLBACK_MESSAGE,
    GenerationContext,
    RagEngine,
    RagEngineError,
    build_system_prompt,
    format_context,
    has_disclaimer,
)


def settings(**overrides) -> Settings:
    base = {
        "supabase_url": "https://test-placeholder.supabase.co",
        "supabase_service_role_key": "test-placeholder",
        "groq_api_key": "test-not-real",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def chunk(section: str = "8.14(2)", **overrides) -> Chunk:
    base = dict(
        id=uuid4(),
        municipality_id="nb_fredericton",
        province_code="NB",
        bylaw_name="Zoning By-law Z-5",
        section_number=section,
        section_title="Uses",
        chunk_content="(a) Permitted Uses\n(1) Child Care Centre\n(b) Conditional Uses\n(1) Kennel",
        page_number=167,
        similarity=0.748,
    )
    base.update(overrides)
    return Chunk(**base)


def ctx(**overrides) -> GenerationContext:
    base = dict(
        municipality_id="nb_fredericton",
        municipality_name="Fredericton",
        province_code="NB",
        language="en",
        bylaw_last_verified_at=None,
    )
    base.update(overrides)
    return GenerationContext(**base)


class StubStream:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    def __aiter__(self):
        async def gen():
            for delta in self._deltas:
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=delta),
                            finish_reason=None,
                        )
                    ]
                )

        return gen()


class StubCompletions:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return StubStream(self.deltas)


def make_engine(deltas: list[str]) -> tuple[RagEngine, StubCompletions]:
    completions = StubCompletions(deltas)
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    return RagEngine(settings(), client=client), completions  # type: ignore[arg-type]


def run(engine, query="Are kennels allowed?", chunks=None, context=None) -> str:
    return asyncio.run(
        engine.generate(query, chunks if chunks is not None else [chunk()], context or ctx())
    )


# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------


def test_missing_groq_key_raises_with_the_free_key_url():
    with pytest.raises(RagEngineError, match="console.groq.com"):
        RagEngine(settings(groq_api_key=None))


# ---------------------------------------------------------------------
#  Context rendering
# ---------------------------------------------------------------------


def test_each_passage_carries_a_ready_made_citation():
    """Rule 3 is mandatory; a model assembling it from parts will slip."""
    rendered = format_context([chunk()], ctx())
    assert "CITE AS: [Fredericton - Zoning By-law Z-5, Section 8.14(2)]" in rendered


def test_passages_are_numbered_and_separated():
    rendered = format_context([chunk("8.1(1)"), chunk("8.2(1)")], ctx())
    assert "[Passage 1]" in rendered and "[Passage 2]" in rendered


def test_empty_context_is_stated_explicitly():
    assert "no matching bylaw passages" in format_context([], ctx())


# ---------------------------------------------------------------------
#  Prompt construction
# ---------------------------------------------------------------------


def test_prompt_names_the_municipality_and_province():
    prompt = build_system_prompt([chunk()], ctx())
    assert "Fredericton, New Brunswick" in prompt


def test_prompt_is_not_restricted_to_atlantic_canada():
    """Phase 0 locked scope to all of Canada."""
    prompt = build_system_prompt([chunk()], ctx(province_code="BC", municipality_name="Kelowna"))
    assert "Atlantic" not in prompt
    assert "British Columbia" in prompt


def test_prompt_carries_the_exact_fallback_wording():
    assert FALLBACK_MESSAGE["en"] in build_system_prompt([chunk()], ctx())


def test_prompt_forbids_mixing_municipalities():
    prompt = build_system_prompt([chunk()], ctx())
    assert "DO NOT make up rules" in prompt
    assert "Every chunk above is from Fredericton only" in prompt


def test_prompt_requires_conflicts_to_be_surfaced():
    assert "flag the conflict explicitly" in build_system_prompt([chunk()], ctx())


def test_prompt_separates_permitted_from_conditional_uses():
    """The corpus contains both under one clause; conflating them misleads."""
    prompt = build_system_prompt([chunk()], ctx())
    assert "Never describe a conditional use as permitted" in prompt


def test_prompt_requests_the_answer_language():
    assert "Answer in French" in build_system_prompt([chunk()], ctx(language="fr"))


# ---------------------------------------------------------------------
#  Disclaimer
# ---------------------------------------------------------------------


def test_unverified_bylaw_says_so_rather_than_printing_a_date():
    """bylaw_last_verified_at is NULL until a human confirms currency."""
    disclaimer = ctx(bylaw_last_verified_at=None).disclaimer()
    assert "NOT yet been verified" in disclaimer
    assert "None" not in disclaimer


def test_verified_bylaw_prints_the_date():
    disclaimer = ctx(bylaw_last_verified_at=date(2026, 9, 7)).disclaimer()
    assert "last verified on 2026-09-07" in disclaimer


def test_french_disclaimer_is_used_for_french_answers():
    assert "Avis" in ctx(language="fr").disclaimer()


def test_disclaimer_is_appended_when_the_model_omits_it():
    """Models drop trailing instructions; this one is not optional."""
    engine, _ = make_engine(["Kennels are a **conditional use**. "])
    answer = run(engine)
    assert has_disclaimer(answer)
    assert DISCLAIMER_UNVERIFIED["en"].strip() in answer


def test_disclaimer_is_not_duplicated_when_the_model_produces_it():
    engine, _ = make_engine(["Answer text.", DISCLAIMER_UNVERIFIED["en"]])
    answer = run(engine)
    assert answer.count("*Disclaimer:") == 1


# ---------------------------------------------------------------------
#  Fallback
# ---------------------------------------------------------------------


def test_no_chunks_returns_the_fallback_without_calling_the_model():
    engine, completions = make_engine(["should not be used"])
    answer = run(engine, chunks=[])
    assert FALLBACK_MESSAGE["en"] in answer
    assert completions.calls == [], "no generation call should be made"


def test_fallback_still_carries_the_disclaimer():
    engine, _ = make_engine([])
    assert has_disclaimer(run(engine, chunks=[]))


def test_french_fallback_is_used_for_french_queries():
    engine, _ = make_engine([])
    answer = run(engine, chunks=[], context=ctx(language="fr"))
    assert FALLBACK_MESSAGE["fr"] in answer


# ---------------------------------------------------------------------
#  Streaming
# ---------------------------------------------------------------------


def test_tokens_are_yielded_incrementally():
    engine, _ = make_engine(["Kennels ", "are ", "conditional."])

    async def collect():
        return [part async for part in engine.stream("q", [chunk()], ctx())]

    parts = asyncio.run(collect())
    # Three model deltas plus the appended disclaimer.
    assert parts[:3] == ["Kennels ", "are ", "conditional."]
    assert len(parts) == 4


def test_model_parameters_come_from_settings():
    engine, completions = make_engine(["ok"])
    run(engine)
    call = completions.calls[0]
    assert call["model"] == settings().llm_model
    assert call["temperature"] == settings().llm_temperature
    assert call["max_tokens"] == settings().llm_max_tokens
    assert call["stream"] is True


def test_user_question_is_sent_separately_from_the_system_prompt():
    engine, completions = make_engine(["ok"])
    run(engine, query="Are kennels allowed in RR-CH?")
    messages = completions.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Are kennels allowed in RR-CH?"}


def test_retrieved_passages_reach_the_system_prompt():
    engine, completions = make_engine(["ok"])
    run(engine)
    assert "Conditional Uses" in completions.calls[0]["messages"][0]["content"]


def test_prompt_demands_ascii_square_brackets_for_citations():
    """gpt-oss-120b emitted full-width 【 】 until told otherwise, which
    would break the Phase 4 CitationCard parser."""
    prompt = build_system_prompt([chunk()], ctx())
    assert "ASCII square brackets" in prompt
    assert "【" in prompt  # named as forbidden, so the model can avoid it


# ---------------------------------------------------------------------
#  Truncation
#
#  The free tier caps output below what a long list of zoning conditions
#  needs. A reply that stops mid-clause reads as a complete answer that
#  merely omits the remaining requirements - and for a setback or a
#  lot-coverage rule, the omitted half is the one that matters.
# ---------------------------------------------------------------------


class StubStreamWithFinish:
    def __init__(self, deltas: list[str], finish: str) -> None:
        self._deltas = deltas
        self._finish = finish

    def __aiter__(self):
        async def gen():
            for delta in self._deltas:
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=delta),
                            finish_reason=None,
                        )
                    ]
                )
            yield types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(content=None),
                        finish_reason=self._finish,
                    )
                ]
            )

        return gen()


def engine_finishing(reason: str, deltas: list[str]):
    class Completions:
        async def create(self, **kwargs):
            return StubStreamWithFinish(deltas, reason)

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    return RagEngine(settings(), client=client)  # type: ignore[arg-type]


def test_truncated_answer_says_so():
    engine = engine_finishing("length", ["The lot must be at least 550 m"])
    answer = run(engine)
    assert "cut short" in answer
    assert "do not treat the list above as complete" in answer


def test_complete_answer_carries_no_truncation_notice():
    engine = engine_finishing("stop", ["Kennels are conditional."])
    assert "cut short" not in run(engine)


def test_truncation_notice_is_localised():
    engine = engine_finishing("length", ["Le lot doit"])
    answer = run(engine, context=ctx(language="fr"))
    assert "interrompue" in answer


def test_truncated_answer_still_gets_the_disclaimer():
    engine = engine_finishing("length", ["partial"])
    assert has_disclaimer(run(engine))
