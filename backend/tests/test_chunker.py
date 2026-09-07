"""Unit tests for app.services.chunker (Phase 2, Step 3)."""

from __future__ import annotations

import pytest

from app.services.chunker import (
    MAX_CHUNK_CHARS,
    ChunkPayload,
    MunicipalityContext,
    build_chunk_content,
    chunk_clauses,
    split_clause_text,
)
from app.services.document_parser import Clause

CONTEXT = MunicipalityContext(
    municipality_id="nb_fredericton",
    municipality_name="Fredericton",
    province_code="NB",
    bylaw_name="Zoning By-law Z-5",
    source_url="https://www.fredericton.ca/z5.pdf",
    language="en",
    document_version="2023-04",
)


def make_clause(**overrides) -> Clause:
    base = dict(
        section_number="8.14(4)",
        section_title="Standards",
        text="(a) Lot Area (MIN)\n(i) Interior Lot: 345 m²",
        page_number=167,
        page_label="8-37",
        part_number="8",
        part_title="Low Density Residential Zones",
        parent_number="8.14",
        parent_title="RURAL RESIDENTIAL - CHATEAU HEIGHTS ZONE",
        amendments=["Z-5.197"],
    )
    base.update(overrides)
    return Clause(**base)


# ---------------------------------------------------------------------
#  Splitting
# ---------------------------------------------------------------------


def test_short_clause_is_not_split():
    assert split_clause_text("(a) Lot Area (MIN)") == ["(a) Lot Area (MIN)"]


def test_empty_clause_yields_nothing():
    assert split_clause_text("   \n  ") == []


def test_long_clause_splits_only_at_subclause_markers():
    body = "\n".join(f"({chr(97 + i)}) " + "rule text " * 40 for i in range(12))
    pieces = split_clause_text(body, limit=1000)

    assert len(pieces) > 1
    # Every piece must begin at a marker, never mid-sentence.
    for piece in pieces:
        assert piece.lstrip().startswith("(")


def test_split_preserves_every_subclause():
    body = "\n".join(f"({chr(97 + i)}) unique-token-{i} " + "filler " * 40 for i in range(10))
    rejoined = "\n".join(split_clause_text(body, limit=800))
    for i in range(10):
        assert f"unique-token-{i}" in rejoined


def test_pieces_respect_the_limit_where_boundaries_allow():
    body = "\n".join(f"({chr(97 + i)}) " + "word " * 30 for i in range(20))
    assert all(len(p) <= 1000 for p in split_clause_text(body, limit=1000))


def test_a_single_oversized_sentence_is_left_intact():
    """Cutting mid-sentence would fabricate a rule; the embedder guards size."""
    body = "x" * 3000
    assert split_clause_text(body, limit=1000) == [body]


def test_stub_tail_is_folded_into_the_previous_piece():
    body = "\n".join(f"({chr(97 + i)}) " + "word " * 60 for i in range(4)) + "\n(z) tiny"
    pieces = split_clause_text(body, limit=700)
    assert pieces[-1].rstrip().endswith("(z) tiny")
    assert len(pieces[-1]) > len("(z) tiny")


# ---------------------------------------------------------------------
#  Contextual payload (Section 4)
# ---------------------------------------------------------------------


def test_payload_carries_province_municipality_and_bylaw():
    content = build_chunk_content(make_clause(), CONTEXT, "body text")
    assert "[Province: NB]" in content
    assert "[Municipality: Fredericton]" in content
    assert "[Bylaw: Zoning By-law Z-5]" in content
    assert "Section 8.14(4) Standards:" in content


def test_payload_repeats_the_parent_zone_heading():
    """"35 % of the lot area" is a different rule in every zone."""
    content = build_chunk_content(make_clause(), CONTEXT, "35 % of the lot area")
    assert "RURAL RESIDENTIAL - CHATEAU HEIGHTS ZONE" in content


def test_split_chunks_are_labelled_as_parts():
    content = build_chunk_content(make_clause(), CONTEXT, "body", part=2, of=3)
    assert "(part 2 of 3)" in content


def test_unsplit_chunk_has_no_part_marker():
    assert "(part" not in build_chunk_content(make_clause(), CONTEXT, "body")


def test_payload_omits_absent_hierarchy_without_printing_none():
    clause = make_clause(parent_number=None, parent_title=None, part_number=None, part_title=None)
    content = build_chunk_content(clause, CONTEXT, "body")
    assert "None" not in content


# ---------------------------------------------------------------------
#  Rows and metadata
# ---------------------------------------------------------------------


def test_chunk_row_matches_the_table_columns():
    payload = chunk_clauses([make_clause()], CONTEXT)[0]
    assert set(payload.as_row()) == {
        "municipality_id",
        "province_code",
        "bylaw_name",
        "section_number",
        "section_title",
        "chunk_content",
        "language",
        "page_number",
        "source_document_version",
        "metadata",
        "chunk_index",
    }


def test_metadata_matches_the_section_4_payload():
    meta = chunk_clauses([make_clause()], CONTEXT)[0].metadata
    assert meta["province"] == "NB"
    assert meta["municipality"] == "nb_fredericton"
    assert meta["section"] == "8.14(4)"
    assert meta["page"] == 167
    assert meta["url"] == "https://www.fredericton.ca/z5.pdf"
    assert meta["document_version"] == "2023-04"
    assert meta["amendments"] == ["Z-5.197"]
    assert meta["page_label"] == "8-37"


def test_absent_optional_metadata_is_omitted_not_nulled():
    clause = make_clause(amendments=[], page_label=None, parent_number=None)
    meta = chunk_clauses([clause], CONTEXT)[0].metadata
    assert "amendments" not in meta
    assert "page_label" not in meta
    assert "parent_section" not in meta


# ---------------------------------------------------------------------
#  Natural key
# ---------------------------------------------------------------------


def test_chunk_index_is_unique_across_clauses_sharing_a_section_number():
    """Z-5 numbers three separate sign tables 6.4; all would collide at 0."""
    clauses = [
        make_clause(section_number="6.4", section_title="COMMERCIAL ZONES", text="a"),
        make_clause(section_number="6.4", section_title="INDUSTRIAL ZONES", text="b"),
        make_clause(section_number="6.4", section_title="RESIDENTIAL ZONES", text="c"),
    ]
    payloads = chunk_clauses(clauses, CONTEXT)
    assert [p.chunk_index for p in payloads] == [0, 1, 2]


def test_natural_key_is_unique_across_a_whole_document():
    clauses = [
        make_clause(section_number="6.4", text="a"),
        make_clause(section_number="6.4", text="b"),
        make_clause(section_number="8.1(1)", text="c"),
    ]
    keys = {
        (p.municipality_id, p.bylaw_name, p.section_number, p.language, p.chunk_index)
        for p in chunk_clauses(clauses, CONTEXT)
    }
    assert len(keys) == 3


def test_split_clause_indexes_run_consecutively():
    long_body = "\n".join(f"({chr(97 + i)}) " + "word " * 60 for i in range(12))
    payloads = chunk_clauses([make_clause(text=long_body)], CONTEXT)
    assert len(payloads) > 1
    assert [p.chunk_index for p in payloads] == list(range(len(payloads)))


def test_empty_clause_produces_no_rows():
    assert chunk_clauses([make_clause(text="  ")], CONTEXT) == []


def test_language_is_stamped_from_context():
    french = MunicipalityContext(
        municipality_id="nb_fredericton",
        municipality_name="Fredericton",
        province_code="NB",
        bylaw_name="Arrêté de zonage Z-5",
        source_url="https://www.fredericton.ca/z5fr.pdf",
        language="fr",
    )
    payload = chunk_clauses([make_clause()], french)[0]
    assert payload.language == "fr"
    assert payload.metadata["language"] == "fr"


@pytest.mark.parametrize("limit", [500, 1200, MAX_CHUNK_CHARS])
def test_no_content_is_lost_at_any_limit(limit):
    body = "\n".join(f"({chr(97 + i)}) marker-{i} " + "filler " * 25 for i in range(15))
    payloads = chunk_clauses([make_clause(text=body)], CONTEXT, limit=limit)
    joined = "\n".join(p.chunk_content for p in payloads)
    for i in range(15):
        assert f"marker-{i}" in joined
