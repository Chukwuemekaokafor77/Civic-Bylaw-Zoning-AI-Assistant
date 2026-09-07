"""Unit tests for app.services.document_parser (Phase 2, Step 2).

Hermetic: no PDF file and no network. Word dictionaries are built with the
geometry measured from Fredericton Zoning By-law Z-5 - left column at
x0=143, right column at x0=372, headings in Arial-Bold 12pt at x0=72 -
so these lock in the specific extraction defects found against the real
document rather than an imagined layout.
"""

from __future__ import annotations

import pytest

from app.services.document_parser import (
    NumberingScheme,
    ParsedPage,
    PageLine,
    _clean_text,
    _definition_term,
    _find_matrix,
    _group_lines,
    _linearise_columns,
    _looks_like_toc,
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
