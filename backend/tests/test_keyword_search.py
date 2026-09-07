"""Unit tests for app.services.keyword_search (Phase 3, Step 2).

Hermetic: the Supabase client is a recording stub. The fusion cases
reproduce measured output from the live 623-chunk Fredericton corpus, so
they pin the behaviour that was actually observed rather than an assumed
one.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.models.schemas import Chunk
from app.services.keyword_search import (
    RPC_NAME,
    KeywordSearch,
    extract_section_hint,
    reciprocal_rank_fusion,
)


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        supabase_url="https://test-placeholder.supabase.co",
        supabase_service_role_key="test-placeholder",
    )


class StubRpc:
    def __init__(self, client: "StubClient", name: str, params: dict) -> None:
        self._client = client
        client.calls.append({"name": name, "params": params})

    async def execute(self):
        return type("R", (), {"data": self._client.rows})()


class StubClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.rows: list[dict] = []

    def rpc(self, name, params):
        return StubRpc(self, name, params)


def make_search() -> tuple[KeywordSearch, StubClient]:
    client = StubClient()
    return KeywordSearch(client, settings()), client  # type: ignore[arg-type]


def rpc_row(section: str = "8.14(4)", rank: float = 1.0) -> dict:
    return {
        "id": str(uuid4()),
        "municipality_id": "nb_fredericton",
        "province_code": "NB",
        "bylaw_name": "Zoning By-law Z-5",
        "section_number": section,
        "section_title": "Standards",
        "chunk_content": "(a) Lot Area (MIN)",
        "language": "en",
        "page_number": 167,
        "metadata": {},
        "rank": rank,
    }


# ---------------------------------------------------------------------
#  Section hint extraction
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What does section 8.14(4) say?", "8.14(4)"),
        ("what are the rules in 7.3(7)?", "7.3(7)"),
        ("tell me about 6.3(1)(a)", "6.3(1)(a)"),
        ("explain 6.3.2 please", "6.3.2"),
        ("what is in 8.14", "8.14"),
        ("what does section 7 cover", "7"),
        ("see Article 4 of the bylaw", "4"),
    ],
)
def test_section_numbers_are_extracted(query, expected):
    assert extract_section_hint(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "can I build a garden suite",
        "how far back does my shed need to be",
        "",
    ],
)
def test_questions_without_a_citation_yield_no_hint(query):
    assert extract_section_hint(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "must be 3 metres from the lot line",
        "I have 2 dwellings on the lot",
        "can I keep 6 hens",
    ],
)
def test_bare_numbers_are_not_read_as_section_numbers(query):
    """Otherwise "3 metres" pins an unrelated Part 3 clause at score 1.00."""
    assert extract_section_hint(query) is None


def test_most_specific_citation_wins():
    """"8.14(4)" must not be truncated to the "8.14" inside it."""
    assert extract_section_hint("what does 8.14(4) require") == "8.14(4)"


# ---------------------------------------------------------------------
#  RPC execution
# ---------------------------------------------------------------------


def run_search(search, query="what does section 8.14(4) say", **kwargs):
    return asyncio.run(search.search(query, "nb_fredericton", **kwargs))


def test_search_calls_the_spec_named_rpc():
    search, client = make_search()
    client.rows = [rpc_row()]
    run_search(search)
    assert client.calls[0]["name"] == RPC_NAME


def test_section_hint_is_derived_and_passed_through():
    search, client = make_search()
    client.rows = [rpc_row()]
    run_search(search)
    assert client.calls[0]["params"]["section_hint"] == "8.14(4)"


def test_explicit_section_hint_overrides_extraction():
    search, client = make_search()
    client.rows = [rpc_row()]
    run_search(search, section_hint="6.3")
    assert client.calls[0]["params"]["section_hint"] == "6.3"


def test_hint_is_null_when_the_question_names_no_section():
    search, client = make_search()
    client.rows = []
    run_search(search, query="can I build a garden suite")
    assert client.calls[0]["params"]["section_hint"] is None


def test_language_defaults_to_configured_rather_than_null():
    """NULL means "any language" in the RPC, breaking the bilingual lock."""
    search, client = make_search()
    client.rows = []
    run_search(search)
    assert client.calls[0]["params"]["target_language"] == "en"


def test_municipality_is_scoped_inside_the_rpc():
    search, client = make_search()
    client.rows = []
    run_search(search)
    assert client.calls[0]["params"]["target_municipality"] == "nb_fredericton"


def test_match_count_defaults_from_settings():
    search, client = make_search()
    client.rows = []
    run_search(search)
    assert client.calls[0]["params"]["match_count"] == settings().keyword_match_count


def test_rows_become_chunks_with_score_seeded_from_rank():
    search, client = make_search()
    client.rows = [rpc_row(rank=0.9)]
    hits = run_search(search)
    assert hits[0].rank == pytest.approx(0.9)
    assert hits[0].score == pytest.approx(0.9)


# ---------------------------------------------------------------------
#  Reciprocal rank fusion
# ---------------------------------------------------------------------


def chunk(section: str, *, similarity=None, rank=None, id_: UUID | None = None) -> Chunk:
    return Chunk(
        id=id_ or uuid4(),
        municipality_id="nb_fredericton",
        province_code="NB",
        bylaw_name="Zoning By-law Z-5",
        section_number=section,
        chunk_content="body",
        similarity=similarity,
        rank=rank,
    )


def test_named_section_outranks_a_merely_similar_clause():
    """Measured live: vector missed 7.3(7) entirely; keyword scored it 1.00."""
    vector = [
        chunk("7.3(10)", similarity=0.66),
        chunk("7.3(2)", similarity=0.66),
        chunk("7.3(3)", similarity=0.66),
    ]
    keyword = [chunk("7.3(7)", rank=1.00)]

    fused = reciprocal_rank_fusion([vector, keyword], limit=4)
    assert fused[0].section_number == "7.3(7)"


def test_named_section_is_promoted_from_third_place():
    """Measured live: vector ranked 8.14(4) third behind its siblings."""
    shared = uuid4()
    vector = [
        chunk("8.14(3)", similarity=0.70),
        chunk("8.14(1)", similarity=0.70),
        chunk("8.14(4)", similarity=0.69, id_=shared),
    ]
    keyword = [chunk("8.14(4)", rank=1.00, id_=shared)]

    fused = reciprocal_rank_fusion([vector, keyword])
    assert fused[0].section_number == "8.14(4)"


def test_agreement_between_retrievers_promotes_a_chunk():
    """The property that makes hybrid better than either half."""
    both = uuid4()
    vector = [chunk("4.1(3)", similarity=0.71, id_=both), chunk("7.2(1)", similarity=0.73)]
    keyword = [chunk("4.1(3)", rank=0.07, id_=both), chunk("7.3(6)", rank=0.11)]

    fused = reciprocal_rank_fusion([vector, keyword])
    assert fused[0].section_number == "4.1(3)"


def test_duplicate_chunks_collapse_to_one_row():
    """Two copies would read to the model as two independent provisions."""
    shared = uuid4()
    vector = [chunk("8.14(4)", similarity=0.69, id_=shared)]
    keyword = [chunk("8.14(4)", rank=1.00, id_=shared)]

    fused = reciprocal_rank_fusion([vector, keyword])
    assert len(fused) == 1


def test_merged_duplicate_keeps_both_signals():
    shared = uuid4()
    fused = reciprocal_rank_fusion(
        [[chunk("8.14(4)", similarity=0.69, id_=shared)],
         [chunk("8.14(4)", rank=1.00, id_=shared)]]
    )
    assert fused[0].similarity == pytest.approx(0.69)
    assert fused[0].rank == pytest.approx(1.00)


def test_weights_shift_the_balance_between_retrievers():
    vector = [chunk("A", similarity=0.7)]
    keyword = [chunk("B", rank=0.1)]

    vector_led = reciprocal_rank_fusion([vector, keyword], weights=[10.0, 1.0])
    keyword_led = reciprocal_rank_fusion([vector, keyword], weights=[1.0, 10.0])
    assert vector_led[0].section_number == "A"
    assert keyword_led[0].section_number == "B"


def test_mismatched_weights_are_rejected():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[chunk("A")], [chunk("B")]], weights=[1.0])


def test_limit_truncates_the_fused_list():
    vector = [chunk(f"8.{i}(1)", similarity=0.7) for i in range(10)]
    assert len(reciprocal_rank_fusion([vector], limit=3)) == 3


def test_empty_inputs_produce_an_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_list_does_not_suppress_the_other():
    vector = [chunk("8.14(4)", similarity=0.7)]
    fused = reciprocal_rank_fusion([vector, []])
    assert [c.section_number for c in fused] == ["8.14(4)"]


def test_score_is_replaced_by_the_fused_value():
    vector = [chunk("8.14(4)", similarity=0.69)]
    fused = reciprocal_rank_fusion([vector])
    # RRF score for a single rank-1 hit is 1/(60+1).
    assert fused[0].score == pytest.approx(1 / 61)
