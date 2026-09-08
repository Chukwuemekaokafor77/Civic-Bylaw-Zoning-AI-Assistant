"""Unit tests for app.services.document_parser (Phase 2, Step 2).

Hermetic: no PDF file and no network. Word dictionaries are built with the
geometry measured from Fredericton Zoning By-law Z-5 - left column at
x0=143, right column at x0=372, headings in Arial-Bold 12pt at x0=72 -
so these lock in the specific extraction defects found against the real
document rather than an imagined layout.
"""

from __future__ import annotations

import re

import pytest

from app.services.document_parser import (
    NumberingScheme,
    SCHEMES,
    ParsedPage,
    PageLine,
    _clean_text,
    _definition_term,
    _find_matrix,
    _group_lines,
    _linearise_columns,
    _line_is_bilingual_prose,
    _looks_like_contents,
    _looks_like_divider,
    _looks_like_term_index,
    _looks_like_toc,
    _table_identity,
    _render_matrix,
    _strip_amendments,
    source_hash,
)

SCHEME = NumberingScheme()

BOLD = "LGGUDB+Arial-BoldMT"
BODY = "SBVKXN+ArialMT"


def word(
    text: str,
    x0: float,
    top: float,
    *,
    font: str = BODY,
    width: float = 6.0,
    height: float = 11.0,
) -> dict:
    """One extract_words() record.

    `top`/`bottom` matter: superscripts are detected from the word box,
    not from a `size` attribute. Requesting `size` from pdfplumber splits
    words wherever the size changes mid-word, which corrupted the zone
    codes in the sign matrix.
    """
    return {
        "text": text,
        "x0": x0,
        "x1": x0 + width * len(text),
        "top": top,
        "bottom": top + height,
        "fontname": font,
    }


def line_of(text: str, x0: float, top: float, *, font: str = BODY) -> list[dict]:
    """A run of words laid out left to right from x0."""
    words = []
    cursor = x0
    for token in text.split():
        w = word(token, cursor, top, font=font)
        words.append(w)
        cursor = w["x1"] + 4.0
    return words


# ---------------------------------------------------------------------
#  Line grouping
# ---------------------------------------------------------------------


def test_words_within_tolerance_form_one_line():
    """Z-5 sets a zone name, number and code at tops 92/93/97."""
    words = [
        word("RURAL", 143, 92),
        word("8.14", 129, 93),
        word("RR-CH", 400, 95),
    ]
    lines = _group_lines(words)
    assert len(lines) == 1
    assert lines[0].text == "8.14 RURAL RR-CH"  # ordered by x, not by top


def test_distant_lines_stay_separate():
    lines = _group_lines([word("first", 72, 100), word("second", 72, 140)])
    assert [ln.text for ln in lines] == ["first", "second"]


def test_starts_bold_reads_the_first_word_font():
    bold = _group_lines(line_of("8.14(4) Standards", 72, 100, font=BOLD))[0]
    plain = _group_lines(line_of("8.3(4)(b) and 8.3(4)(c).", 171, 100))[0]
    assert bold.starts_bold is True
    assert plain.starts_bold is False


# ---------------------------------------------------------------------
#  Amendment markers
# ---------------------------------------------------------------------


def test_amendment_marker_is_lifted_out_as_metadata():
    lines = _group_lines(line_of("(4) Keeping of Hens", 143, 380) + [word("Z-5.197", 530, 380)])
    cleaned, amendments = _strip_amendments(lines, SCHEME)
    assert amendments == ["Z-5.197"]
    assert "Z-5.197" not in cleaned[0].text


def test_amendment_only_line_is_dropped_not_left_blank():
    lines = _group_lines([word("Z-5.82", 530, 200)])
    cleaned, amendments = _strip_amendments(lines, SCHEME)
    assert cleaned == []
    assert amendments == ["Z-5.82"]


# ---------------------------------------------------------------------
#  Two-column use lists — the highest-stakes extraction defect
# ---------------------------------------------------------------------


def build_use_list() -> list:
    """The 8.14(2) Uses layout: two columns, left one row longer."""
    words: list[dict] = []
    words += line_of("(a) Permitted Uses", 143, 308)
    words += line_of("(b) Conditional Uses", 358, 308)
    words += line_of("(1) Child Care Centre", 143, 326)
    words += line_of("(1) Kennel", 372, 326)
    words += line_of("(2) Home Occupation", 143, 343)
    words += line_of("(2) Studio - Artisan", 372, 343)
    words += line_of("(3) Single Detached Dwelling", 143, 361)
    words += line_of("(4) Keeping of Hens", 143, 380)
    return _group_lines(words)


def test_permitted_and_conditional_uses_are_not_interleaved():
    """A kennel is a conditional use here; reading across makes it permitted."""
    result = _linearise_columns(build_use_list(), 612.0, SCHEME)
    text = [ln.text for ln in result]

    assert text == [
        "(a) Permitted Uses",
        "(1) Child Care Centre",
        "(2) Home Occupation",
        "(3) Single Detached Dwelling",
        "(4) Keeping of Hens",
        "(b) Conditional Uses",
        "(1) Kennel",
        "(2) Studio - Artisan",
    ]
    # The two lists must not be adjacent on one line anywhere.
    assert not any("Child Care Centre (1) Kennel" in t for t in text)


def test_hanging_indent_marker_stays_with_its_column():
    """"(b)" is set ~14pt left of the text it labels."""
    result = _linearise_columns(build_use_list(), 612.0, SCHEME)
    text = [ln.text for ln in result]
    assert "(b) Conditional Uses" in text
    assert "(a) Permitted Uses (b)" not in text


def test_left_column_runs_on_past_the_last_two_column_row():
    result = _linearise_columns(build_use_list(), 612.0, SCHEME)
    text = [ln.text for ln in result]
    # (3) and (4) have no right-hand counterpart but are still left column,
    # so they must precede the Conditional list.
    assert text.index("(4) Keeping of Hens") < text.index("(b) Conditional Uses")


def test_wrapped_prose_is_never_split_into_columns():
    """Prose crosses the page continuously and has no gutter."""
    words = line_of(
        "All uses shall comply with the Regulations Applying to All Uses and more", 72, 432
    )
    words += line_of("Applying to Residential Uses (Section 7).", 72, 446)
    result = _linearise_columns(_group_lines(words), 612.0, SCHEME)
    assert [ln.text for ln in result] == [
        "All uses shall comply with the Regulations Applying to All Uses and more",
        "Applying to Residential Uses (Section 7).",
    ]


def build_zone_standards() -> list:
    """The 8.3(4) Standards layout: label at the margin, value far right.

    Geometrically this is indistinguishable from a two-column list - the
    gap between "Interior Lot:" and "345" is 163pt, far wider than any
    gutter - but the right side is the value OF the row, not a column.
    """
    words: list[dict] = []
    words += line_of("(a) Lot Area (MIN)", 117, 473, font=BOLD)
    words += line_of("(i) Interior Lot:", 144, 501)
    words += [word("345", 388, 501), word("m", 410, 501)]
    words += line_of("(ii) Corner Lot", 144, 521)
    words += [word("480", 388, 521), word("m", 410, 521)]
    words += line_of("(b) Lot Frontage (MIN)", 117, 550, font=BOLD)
    words += line_of("(i) Interior Lot:", 144, 577)
    words += [word("11.5", 388, 577), word("metres", 412, 577)]
    return _group_lines(words)


def test_measurements_stay_on_the_row_they_belong_to():
    """Detaching values strands every setback from the rule it governs."""
    result = _linearise_columns(build_zone_standards(), 612.0, SCHEME)
    text = [ln.text for ln in result]

    assert "(i) Interior Lot: 345 m" in text
    assert "(ii) Corner Lot 480 m" in text
    assert "(i) Interior Lot: 11.5 metres" in text
    # The failure this prevents: labels with no numbers, and a loose block
    # of numbers that could be paired with any of them.
    assert "(i) Interior Lot:" not in text
    assert "345 m" not in text


def test_value_column_is_distinguished_from_a_real_list_column():
    """The discriminator is whether the right side has its own markers."""
    uses = _linearise_columns(build_use_list(), 612.0, SCHEME)
    standards = _linearise_columns(build_zone_standards(), 612.0, SCHEME)

    # Uses: right side carries "(b)", "(1)", "(2)" -> reordered.
    assert [ln.text for ln in uses][-1] == "(2) Studio - Artisan"
    # Standards: right side is bare measurements -> left untouched.
    assert [ln.text for ln in standards] == [ln.text for ln in build_zone_standards()]


def test_single_column_page_is_returned_unchanged():
    lines = _group_lines(line_of("(a) Lot Area (MIN)", 143, 497))
    assert _linearise_columns(lines, 612.0, SCHEME) == lines


# ---------------------------------------------------------------------
#  Permission matrix
# ---------------------------------------------------------------------


ZONES = ["LC", "NC", "DC", "RC", "OC", "RLF"]


def build_matrix() -> list:
    words: list[dict] = []
    # Header: wrapped legend text then the trailing run of zone codes.
    words += [word("Development", 60, 130), word("Agreement", 110, 130)]
    for i, zone in enumerate(ZONES):
        words.append(word(zone, 190 + i * 24, 130, width=4.0))
    words += line_of("P = Permitted", 60, 150)
    words += line_of("SD = Sign Districts", 60, 162)
    words += [word("CANOPY", 72, 176, font=BOLD)]
    words += [word("6.4(1)", 98, 190)]
    for i in range(4):
        words.append(word("P", 190 + i * 24, 190, width=4.0))
    words.append(word("SD", 190 + 4 * 24, 190, width=4.0))
    words += [word("6.4(2)(a)", 98, 210)]
    return _group_lines(words)


def test_matrix_header_survives_wrapped_legend_text():
    matrix = _find_matrix(build_matrix(), SCHEME)
    assert matrix is not None
    assert [w["text"] for w in matrix.columns.words] == ZONES


def test_matrix_cells_are_mapped_to_their_zone_columns():
    matrix = _find_matrix(build_matrix(), SCHEME)
    rendered = _render_matrix(matrix, {"P": "Permitted", "SD": "Sign Districts"})
    body = "\n".join(rendered)
    assert "CANOPY 6.4(1) - Permitted: LC, NC, DC, RC; Sign Districts: OC." in body
    # The failure this prevents: a bare run of symbols with no zone names.
    assert "P P P P" not in body


def test_empty_matrix_row_is_reported_as_absence_not_prohibition():
    matrix = _find_matrix(build_matrix(), SCHEME)
    rendered = _render_matrix(matrix, {})
    assert "CANOPY 6.4(2)(a) - no entry for any listed zone." in rendered
    assert "not permitted" not in " ".join(rendered).lower()


def test_banner_line_is_consumed_so_it_cannot_leak_into_a_clause():
    matrix = _find_matrix(build_matrix(), SCHEME)
    assert any(ln.text == "CANOPY" for ln in matrix.consumed)


def test_plain_text_page_has_no_matrix():
    lines = _group_lines(line_of("The RR-CH Zone accommodates rural residential", 72, 158))
    assert _find_matrix(lines, SCHEME) is None


# ---------------------------------------------------------------------
#  Definitions
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        ("Convenience Store means a use not exceeding 300 square metres", "Convenience Store"),
        ("Utilities means a use where energy and electricity", "Utilities"),
        ("Garden Suite means a self-contained dwelling", "Garden Suite"),
    ],
)
def test_definition_term_is_the_text_before_means(body, expected):
    assert _definition_term(body) == expected


@pytest.mark.parametrize(
    "body,expected",
    [
        ("« abri d'auto » Garage privé attenant", "abri d'auto"),
        ("« abattoir » Établissement où l'on abat", "abattoir"),
    ],
)
def test_french_definition_term_comes_from_the_guillemets(body, expected):
    """The title is the highest-weighted field in the full-text index."""
    assert _definition_term(body) == expected


def test_definitions_chapter_is_recognised_in_french():
    """"Définitions" must take the same branch as "Definitions"."""
    assert SCHEME.is_definitions_part("Définitions") is True
    assert SCHEME.is_definitions_part("Definitions") is True
    assert SCHEME.is_definitions_part("Parking Standards") is False


def test_french_running_header_carries_a_partie_prefix():
    """French pages read "PARTIE I Section 3 Définitions"."""
    match = SCHEME.running_header.match("PARTIE I Section 3 Définitions")
    assert match is not None
    assert match.group(1) == "3"
    assert match.group(2) == "Définitions"


def test_english_running_header_still_matches():
    match = SCHEME.running_header.match("Section 8 Low Density Residential Zones RR-CH")
    assert match is not None
    assert match.group(1) == "8"


def test_definition_term_falls_back_when_means_is_absent():
    term = _definition_term("Some entry that does not follow the convention at all here")
    assert term.startswith("Some entry that does not")


# ---------------------------------------------------------------------
#  Table of contents
# ---------------------------------------------------------------------


def make_page(lines: list[str]) -> ParsedPage:
    return ParsedPage(
        page_number=1,
        lines=[PageLine(text, False) for text in lines],
        page_label=None,
        part_number=None,
        part_title=None,
        amendments=[],
    )


def test_contents_page_is_recognised_by_its_page_references():
    page = make_page(
        [
            "2.1 OPERATION 2-1",
            "2.1(1) Powers of the Development Officer 2-1",
            "2.2 INTERPRETATION 2-7",
            "2.3 ZONES 2-8",
        ]
    )
    assert _looks_like_toc(page) is True


def test_body_page_is_not_mistaken_for_contents():
    page = make_page(
        [
            "No development permit shall pertain to more than one lot.",
            "Upon receipt of a complete application including payment of the fee,",
            "the Development Officer shall issue a development permit.",
        ]
    )
    assert _looks_like_toc(page) is False


# ---------------------------------------------------------------------
#  Text normalisation and hashing
# ---------------------------------------------------------------------


def test_superscript_is_reattached_to_its_unit():
    """"345 m 2" is neither the unit a resident reads nor a searchable term."""
    words = [
        word("345", 388, 501),
        word("m", 410, 501),
        # Measured from Z-5 p135: shorter and sitting on a raised baseline.
        word("2", 419, 500.6, height=6.41),
    ]
    assert _group_lines(words)[0].text == "345 m²"


def test_same_size_digit_is_not_treated_as_a_superscript():
    words = [word("30", 144, 653), word("2", 170, 653)]
    assert _group_lines(words)[0].text == "30 2"


def test_a_small_digit_on_the_same_baseline_is_not_a_superscript():
    """Height alone would misread "case 2"; the raised baseline is required."""
    words = [
        word("case", 144, 653),
        # Shorter box, but bottom-aligned with its neighbour.
        {**word("2", 180, 657, height=7.0), "bottom": 664.0},
    ]
    assert _group_lines(words)[0].text == "case 2"


def test_superscript_detection_tolerates_a_missing_box():
    plain = [
        {"text": "345", "x0": 388, "x1": 400, "top": 501, "fontname": BODY},
        {"text": "2", "x0": 410, "x1": 416, "top": 501, "fontname": BODY},
    ]
    assert _group_lines(plain)[0].text == "345 2"


def test_clean_text_closes_the_gap_before_stray_punctuation():
    assert _clean_text("shall be the minimum lot area .") == "shall be the minimum lot area."
    assert _clean_text("From a front property line :") == "From a front property line:"


def test_clean_text_collapses_spacing_without_touching_wording():
    assert _clean_text("35 %  of   the lot\n\n\n\narea") == "35 % of the lot\n\narea"


def test_clean_text_preserves_accents_and_symbols():
    """French definitions and metric operators must survive intact."""
    assert _clean_text("services d’utilité publique ≥ 20 m2") == "services d’utilité publique ≥ 20 m2"


def test_source_hash_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"bylaw content")
    b.write_bytes(b"bylaw content")
    assert source_hash(a) == source_hash(b)

    b.write_bytes(b"bylaw content amended")
    assert source_hash(a) != source_hash(b)


# ---------------------------------------------------------------------
#  Bilingual documents (Moncton)
# ---------------------------------------------------------------------

MONCTON = SCHEMES["moncton"]


@pytest.mark.parametrize(
    "line, term",
    [
        (
            "“ bicycle parking space ” means a slot in a rack",
            "bicycle parking space",
        ),
        ("“cemetery” means land used for internment", "cemetery"),
        # A qualifier may sit between the term and the verb.
        ("“ city ”, when used alone, means the geographic area", "city"),
    ],
)
def test_quoted_definition_needs_no_bold(line, term):
    """Moncton does not set its defined terms in bold; the quotes mark them."""
    found = MONCTON.definition_quoted.match(line)
    assert found is not None
    assert found.group(1).strip() == term


@pytest.mark.parametrize(
    "line",
    [
        # A continuation line, not a definition: read as one, it opens a
        # clause that swallows the rest of the chapter.
        "which an adult sized bicycle may be secured by means of",
        "copy used for the advertisement of goods or services.",
        # A quoted sentence is not a defined term.
        "“No parking.” The sign means the driver may not stop",
    ],
)
def test_prose_is_not_read_as_a_quoted_definition(line):
    assert MONCTON.definition_quoted.match(line) is None


def test_french_definition_has_no_defining_verb():
    """The guillemets are the whole marker: no "means" to anchor on."""
    found = MONCTON.definition_guillemet.match(
        "« arbre de rue » Arbre à planter entre la limite du lot"
    )
    assert found is not None
    assert found.group(1).strip() == "arbre de rue"


def test_french_cross_reference_is_not_a_definition():
    """A lowercase continuation marks a reference, not a defined term."""
    assert (
        MONCTON.definition_guillemet.match(
            "« arbre de rue » de l’arrêté ne s’applique pas"
        )
        is None
    )


@pytest.mark.parametrize(
    "line, number",
    [
        ("111 (1) Minimum yard requirements do not apply", "111(1)"),
        # A section whose whole content is one paragraph carries no
        # subsection number.
        ("142 In accordance with section 7, Table 12.2 identifies", "142"),
        # pdfplumber splits "82.1" where the glyph spacing changes.
        ("82. 1 (1) Despite Table 14.1, no lot containing", "82.1(1)"),
    ],
)
def test_moncton_section_numbers_survive_the_layout(line, number):
    found = MONCTON.clause.match(line)
    assert found is not None
    assert re.sub(r"\s+", "", found.group(1)) == number


def test_two_columns_of_prose_are_split_by_language():
    words = line_of("No development shall be", 60, 100)
    words += line_of("Les aménagements ne sont", 330, 100)
    assert _line_is_bilingual_prose(words, 316.0) is True


def test_table_row_is_never_split_by_language():
    """Severing a row would strand a setback from the zone it governs."""
    words = [
        word("R-1A", 60, 100),
        word("558", 200, 100),
        word("460", 340, 100),
        word("460", 460, 100),
    ]
    assert _line_is_bilingual_prose(words, 316.0) is False


def test_indented_section_number_is_not_a_column_boundary():
    """Both languages indent the number, leaving a wide gap after it.

    Counting that gap as a column boundary makes the line look like a
    table row, and the two languages stay interleaved in the text.
    """
    words = line_of("158", 60, 100) + line_of("No development permitted", 100, 100)
    words += line_of("158", 330, 100) + line_of("Les aménagements permis", 370, 100)
    assert _line_is_bilingual_prose(words, 316.0) is True


def test_contents_page_without_page_numbers_is_recognised():
    """Moncton's contents carry no page references for _looks_like_toc."""
    page = make_page(
        [
            "PART 16 - DOWNTOWN ZONES",
            "153 Table 16.1 Downtown zones use table",
            "154 Table 16.2 Downtown zones secondary use table",
            "155 Table 16.3 Downtown zones lot requirements table",
            "PART 17 - RURAL AND MANUFACTURED DWELLING ZONES",
            "156 Table 17.1 Rural and manufactured dwelling zones use table",
            "PART 18 - TOURISM ZONE",
            "159 Table 18.1 Tourism zone use table",
            "160 Table 18.2 Tourism zone secondary use table",
            "161 Table 18.3 Tourism zone lot requirements table",
            "162 Integrated Development",
            "163 Conditional agreements carried over",
            "164 Previous approvals",
        ]
    )
    assert _looks_like_contents(page) is True


def test_page_of_provisions_is_not_mistaken_for_contents():
    page = make_page(
        [
            "111 (1) Minimum yard requirements do not apply on the side",
            "of the lot where a commercial or industrial use abuts a",
            "railway right-of-way, but section 110 applies, with the",
            "necessary modifications, to preserve the sight triangle at",
            "the intersection of the railway and the street.",
            "111 (2) Where a new residential development abuts a railway",
            "right-of-way, a minimum 30 metre setback shall be",
            "maintained between the railway right-of-way and a main",
            "building.",
            "Reduced frontage on a curve",
            "112 (1) Where a lot fronts on the outside of a curve, the",
            "minimum frontage may be reduced by up to 20 percent.",
            "112 (2) Subsection (1) does not apply in the MD Zone.",
        ]
    )
    assert _looks_like_contents(page) is False


def test_term_index_is_not_read_as_the_definitions_section():
    """The index pairs terms with translations; it states no rule.

    Parsed, it becomes a 10,000-character clause cited as section 1 -
    the citation the real section 1 already carries.
    """
    page = make_page(
        [
            "accessory building – bâtiment accessoire",
            "accessory use – usage accessoire",
            "additional dwelling unit – logement supplémentaire",
            "adult cabaret – cabaret pour adultes",
            "borrow pit – banc d’emprunt",
            "building – bâtiment",
            "cemetery – cimetière",
            "city – ville",
            "dwelling unit – logement",
            "garden suite – pavillon-jardin",
            "lot – lot",
            "sign – enseigne",
            "zone – zone",
        ]
    )
    assert _looks_like_term_index(page) is True


def test_definitions_section_is_not_mistaken_for_the_index():
    page = make_page(
        [
            "“ garden suite ” means a self-contained dwelling unit that",
            "is accessory to a single unit dwelling on the same lot.",
            "“ grade ” means the average level of finished ground",
            "adjoining a building at all exterior walls.",
            "“ height ” means the vertical distance between grade and",
            "the highest point of the roof surface.",
            "“ lot ” means a parcel of land described in a deed or shown",
            "on a registered subdivision plan.",
            "“ sign ” means any device, structure or medium used to",
            "convey information visually.",
            "“ street ” means a public highway vested in the City.",
            "“ yard ” means an open space on the same lot as a building.",
            "“ zone ” means an area of the City shown on Schedule A.",
        ]
    )
    assert _looks_like_term_index(page) is False


# ---------------------------------------------------------------------
#  Saint John: two heading forms, dot leaders, tab-index dividers
# ---------------------------------------------------------------------

SAINT_JOHN = SCHEMES["saintjohn"]


@pytest.mark.parametrize(
    "line, number, title",
    [
        ("4.2(5) PARKING LOT STANDARDS", "4.2(5)", "PARKING LOT STANDARDS"),
        # The title-case form. Recognising only the first left one clause
        # running from parking standards to signs, thirty pages later.
        ("4.4 Drive-Thru Facilities", "4.4", "Drive-Thru Facilities"),
        ("15.3 Trinity Royal Street Wall", "15.3", "Trinity Royal Street Wall"),
    ],
)
def test_saint_john_heads_a_section_either_way(line, number, title):
    found = SAINT_JOHN.clause.match(line)
    assert found is not None
    assert found.group(1) == number
    assert found.group(2) == title


@pytest.mark.parametrize(
    "line",
    [
        # A bold table cell carrying a section-shaped number. Read as a
        # heading, it opens a clause numbered 0.25.
        "0.25 square metres for each face",
        "0.5 square metres total of all faces",
    ],
)
def test_bold_table_cell_is_not_a_saint_john_heading(line):
    assert SAINT_JOHN.clause.match(line) is None


def test_dot_leader_contents_page_is_recognised():
    """Saint John's contents run a dot leader out to the page number."""
    page = make_page(
        [
            "SCHEDULE A: ZONING MAP..................................... 247",
            "SCHEDULE B: FEES........................................... 251",
            "SCHEDULE C: UPTOWN PARKING EXEMPTION AREA.................. 252",
            "SCHEDULE D: INTENSIFICATION AREAS.......................... 255",
        ]
    )
    assert _looks_like_toc(page) is True


def test_tab_index_divider_page_is_skipped():
    """A part divider lists every part and states no rule.

    Left in, it appends "Residential Zones 10" to whichever clause was
    open - which is how the sidebar ended up inside fourteen of
    Fredericton's clauses.
    """
    page = make_page(
        [
            "Administration 1",
            "Zones and Administration 2",
            "Definitions 3",
            "General Provisions: Access, Parking, and Loading 4",
            "General Provisions: Landscaping and Amenity Space 6",
            "General Provisions: Signs 7",
            "General Provisions: Other Standards 8",
            "Residential Zones 10",
            "Commercial Zones 11",
            "Industrial Zones 12",
            "Community Facility Zones 13",
            "Other Zones 14",
        ]
    )
    assert _looks_like_divider(page) is True


def test_page_of_provisions_is_not_mistaken_for_a_divider():
    page = make_page(
        [
            "(a) A parking lot involving five or more parking spaces located",
            "on a lot in the Primary Development Area shall be developed and",
            "maintained with a paved surface enclosed with permanent curbing.",
            "(b) A parking lot involving five or more parking spaces located",
            "outside of the Primary Development Area shall be developed and",
            "maintained with a paved surface.",
            "(c) Any storey above the maximum street wall height shall step",
            "back at a minimum depth of 3 metres away from the street facade.",
            "Maximum Height: 12",
            "Minimum Side Yard: 3",
        ]
    )
    assert _looks_like_divider(page) is False


def test_schedule_page_is_cited_by_its_caption():
    """A schedule is a map, not part of the clause flow.

    Its caption is how the provisions refer to it ("as delineated by
    Schedule C"), so that is what it is cited as.
    """
    page = make_page(
        [
            "Schedule K: Spruce Lake Industrial (SLI) Zone Setbacks",
            "[2025, C.P. 111-196]",
        ]
    )
    number, title = _table_identity(page)
    assert number == "Schedule K"
    assert title == "Spruce Lake Industrial (SLI) Zone Setbacks"


def test_table_caption_still_wins_over_a_schedule_reference():
    page = make_page(
        [
            "TABLE 12.3 RESIDENTIAL ZONES LOT REQUIREMENTS TABLE",
            "as delineated by Schedule C: Uptown Parking Exemption Area",
        ]
    )
    number, _ = _table_identity(page)
    assert number == "Table 12.3"
