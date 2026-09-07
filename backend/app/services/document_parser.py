"""PDF bylaw parsing and clause extraction (Phase 2, Step 2).

Turns a municipal zoning bylaw PDF into an ordered list of `Clause` records
carrying the section number, title, page, and hierarchy needed for the
Section 5 citation format `[Municipality - Bylaw, Section X.Y]`.

Proven against Fredericton Zoning By-law Z-5 (338 pages). The numbering
rules are supplied by a `NumberingScheme` rather than hard-coded, because
Section 4 requires the same engine to handle the other Atlantic numbering
conventions (`Part VI`, `12(1)(a)`) as those municipalities are onboarded.

Three properties of real bylaw PDFs drive the design:

1. Multi-column use lists. On zone pages, Permitted Uses and Conditional
   Uses sit in side-by-side columns. `extract_text()` reads across them and
   produces "(1) Child Care Centre - Small (1) Kennel", which invites the
   model to report a conditional use as permitted - a confidently wrong
   answer about what someone may build. Columns are therefore detected
   geometrically and linearised one column at a time.

2. Running headers and page labels. Every body page repeats a header
   ("Section 8 Low Density Residential Zones RR-CH") and ends with a
   chapter-relative label ("8-37"). Both would otherwise land mid-clause.
   The label is captured as `page_label`: it is what a resident sees
   printed on the page, and differs from the PDF page index (page 167 is
   labelled 8-37), so citations need both.

3. Inline amendment markers. Amending bylaw numbers ("Z-5.197") are set in
   the right margin of the clause they changed. They are pulled out as
   metadata before column detection - both because they are genuinely
   useful provenance and because a margin marker would otherwise be
   mistaken for a right-hand column.
"""

from __future__ import annotations

import collections
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------
#  Layout constants (PDF points; 612pt = US Letter width)
# ---------------------------------------------------------------------

# Words whose baselines differ by less than this belong to one visual line.
# Fredericton sets a zone name, its number and its code at tops 92/93/97 —
# one row to a reader, three to a naive grouper.
LINE_Y_TOLERANCE = 4.0

# Whitespace wide enough to be a column gutter rather than word spacing.
GUTTER_MIN_WIDTH = 30.0

# A gutter must fall in the middle of the page. Anything further out is a
# margin marker or a hanging indent, not a column break.
GUTTER_SEARCH_MIN_RATIO = 0.35
GUTTER_SEARCH_MAX_RATIO = 0.70

# Right-column starts must agree within this to count as the same column.
COLUMN_X_TOLERANCE = 4.0

# A single stray gap is word spacing; a repeated one is a column.
MIN_COLUMN_LINES = 2

# A list marker opening a line: "(a)", "(iii)", "(A)", "(12)".
LIST_MARKER = re.compile(r"^\((?:[a-z]{1,3}|[A-Z]{1,3}|\d{1,3})\)\s*\S")

# A glyph set below this fraction of the preceding one is a superscript.
SUPERSCRIPT_SIZE_RATIO = 0.8

SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _is_superscript(word: dict, previous: dict) -> bool:
    size, prior = word.get("size"), previous.get("size")
    if not size or not prior:
        return False
    return size < prior * SUPERSCRIPT_SIZE_RATIO


def _strip_accents(text: str) -> str:
    """Fold accented characters to their base letters, for matching only.

    Never applied to stored bylaw text - only to headings being classified,
    so "Définitions" and "Definitions" take the same branch.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass(frozen=True)
class NumberingScheme:
    """Regexes describing one municipality's clause numbering.

    Fredericton's defaults are below. Saint John and CBRM get their own
    schemes as they are onboarded rather than edits to this one.
    """

    # "8.14(4) Standards" — the citable clause level.
    clause: re.Pattern[str] = re.compile(r"^(\d+\.\d+\(\d+[a-z]?\))\s*(.*)$")
    # "2.1 OPERATION" — the subsection heading above clauses.
    subsection: re.Pattern[str] = re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Z0-9 &/,'’()\-\.]{3,})$")
    # "(203) Utilities means ..." — Section 3 definition entries.
    definition: re.Pattern[str] = re.compile(r"^\((\d+)\)\s+(.+)$")
    # Running header: "Section 8 Low Density Residential Zones RR-CH", or
    # its French form "PARTIE I Section 3 Définitions". The French edition
    # prefixes the part in Roman numerals, so anchoring on "Section" alone
    # leaves every French page unattributed - and an unattributed page is
    # also an undetected definitions page.
    running_header: re.Pattern[str] = re.compile(
        r"^(?:(?:PART|PARTIE)\s+[IVXLC\d]+\s+)?Section\s+(\d+)\s+(.*)$",
        re.IGNORECASE,
    )
    # Printed page label: "8-37".
    page_label: re.Pattern[str] = re.compile(r"^(\d+)\s*[-–]\s*(\d+)$")
    # Amending bylaw marker set in the margin: "Z-5.197".
    amendment: re.Pattern[str] = re.compile(r"^[A-Z]{1,3}-\d+(?:\.\d+)+$")
    # Zone code used as a matrix column header: "LC", "COR-1", "MX-2".
    zone_code: re.Pattern[str] = re.compile(r"^[A-Z]{1,5}(?:-\d+)?$")
    # A row label in a permission matrix: "6.4(2)(a)".
    matrix_row: re.Pattern[str] = re.compile(r"^\d+\.\d+(?:\([0-9a-zA-Z]+\))+$")
    # Legend entry: "P = Permitted".
    legend: re.Pattern[str] = re.compile(r"^([A-Z]{1,3})\s*=\s*(.+)$")

    def is_definitions_part(self, part_title: str | None) -> bool:
        """Whether this part is the definitions chapter, in either language.

        Compared with accents stripped: the French edition heads the
        chapter "Définitions", and a plain substring test for "definition"
        never matches it. Missing it does not fail loudly - it silently
        appends the entire definitions chapter to the previous clause,
        which is how the English parse produced one 58,000-character blob
        before this branch existed.
        """
        if not part_title:
            return False
        return "definition" in _strip_accents(part_title).lower()


# Symbols that appear as matrix cells. Anything else on a matrix row means
# the line is prose and must not be read as a table.
MATRIX_CELL_TOKENS = {"P", "SD", "DA", "C", "X", "A", "-", "•"}

# Column centres closer than this belong to the same matrix column.
MATRIX_COLUMN_TOLERANCE = 14.0

# Below this many columns it is not a matrix, just a short line.
MIN_MATRIX_COLUMNS = 5


@dataclass
class Clause:
    """One citable unit of bylaw text."""

    section_number: str
    section_title: str
    text: str
    page_number: int
    page_label: str | None = None
    part_number: str | None = None
    part_title: str | None = None
    parent_number: str | None = None
    parent_title: str | None = None
    amendments: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class _Line:
    top: float
    words: list[dict]

    @property
    def text(self) -> str:
        """Line text, with superscripts reattached to the token they modify.

        Area limits are set as "m" followed by a 6.4pt "2" - a real
        superscript that pdfplumber reports as its own word. Joined with a
        space it reads "345 m 2", which is neither the unit a resident
        recognises nor a term a keyword search for "m2" or "square metres"
        will match. Rejoined it is "345 m²".
        """
        parts: list[str] = []
        for index, word in enumerate(self.words):
            token = word["text"]
            if index and token.isdigit() and _is_superscript(word, self.words[index - 1]):
                parts[-1] += token.translate(SUPERSCRIPT_DIGITS)
                continue
            parts.append(token)
        return " ".join(parts)

    @property
    def x0(self) -> float:
        return min(w["x0"] for w in self.words)

    @property
    def starts_bold(self) -> bool:
        """Whether the line opens in a bold face.

        This is what separates a real heading from a wrapped cross
        reference. Fredericton sets headings in Arial-Bold 12pt at the
        left margin, while body text that happens to begin with a section
        number - "8.3(4)(b) and 8.3(4)(c)." continuing a sentence from the
        previous line - is regular Arial 11pt. Matching on the number
        alone invents clauses that do not exist, truncates the real clause
        mid-sentence, and issues two different citations for one number.
        """
        if not self.words:
            return False
        return "bold" in str(self.words[0].get("fontname", "")).lower()


# ---------------------------------------------------------------------
#  Line assembly
# ---------------------------------------------------------------------


def _group_lines(words: list[dict], tolerance: float = LINE_Y_TOLERANCE) -> list[_Line]:
    """Cluster words into visual lines, tolerating small baseline drift."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[_Line] = []
    bucket: list[dict] = [ordered[0]]
    anchor = ordered[0]["top"]

    for word in ordered[1:]:
        if abs(word["top"] - anchor) <= tolerance:
            bucket.append(word)
        else:
            lines.append(_Line(anchor, sorted(bucket, key=lambda w: w["x0"])))
            bucket = [word]
            anchor = word["top"]

    lines.append(_Line(anchor, sorted(bucket, key=lambda w: w["x0"])))
    return lines


# ---------------------------------------------------------------------
#  Column handling
# ---------------------------------------------------------------------


def _strip_amendments(
    lines: list[_Line],
    scheme: NumberingScheme,
) -> tuple[list[_Line], list[str]]:
    """Remove margin amendment markers, returning them as metadata.

    Done before column detection: a marker in the right margin looks
    exactly like a one-word right-hand column.
    """
    found: list[str] = []
    cleaned: list[_Line] = []

    for line in lines:
        keep = []
        for word in line.words:
            if scheme.amendment.match(word["text"]):
                found.append(word["text"])
            else:
                keep.append(word)
        if keep:
            cleaned.append(_Line(line.top, keep))

    # Preserve first-seen order without duplicates.
    return cleaned, list(dict.fromkeys(found))


def _gutter_start(line: _Line, low: float, high: float) -> float | None:
    """x where this line's own gutter-sized gap ends, if it has one.

    This is the discriminator between a columnar line and prose. Prose
    wraps continuously across the page, so it has no interior gap this
    wide; a two-column row always does. Keying region detection off the
    gap - rather than off "has a word past x" - is what stops a wrapped
    sentence from being torn in half and reassembled out of order.
    """
    for left, right in zip(line.words, line.words[1:]):
        if right["x0"] - left["x1"] >= GUTTER_MIN_WIDTH and low <= right["x0"] <= high:
            return right["x0"]
    return None


def _find_column_boundary(lines: list[_Line], page_width: float) -> float | None:
    """Approximate gutter x agreed on by at least MIN_COLUMN_LINES lines."""
    low = page_width * GUTTER_SEARCH_MIN_RATIO
    high = page_width * GUTTER_SEARCH_MAX_RATIO

    starts: collections.Counter[float] = collections.Counter()
    for line in lines:
        start = _gutter_start(line, low, high)
        if start is not None:
            # Quantise so near-identical starts land in one bucket.
            starts[round(start / COLUMN_X_TOLERANCE) * COLUMN_X_TOLERANCE] += 1

    if not starts:
        return None

    candidate, hits = starts.most_common(1)[0]
    if hits < MIN_COLUMN_LINES:
        return None
    return candidate


def _entirely_one_side(line: _Line, boundary: float) -> bool:
    """True if the line sits wholly in one column (never straddling it)."""
    return all(w["x1"] <= boundary for w in line.words) or all(
        w["x0"] >= boundary for w in line.words
    )


def _linearise_columns(
    lines: list[_Line],
    page_width: float,
    scheme: NumberingScheme,
) -> list[_Line]:
    """Re-order a two-column region into left column then right column."""
    approx = _find_column_boundary(lines, page_width)
    if approx is None:
        return lines

    low = page_width * GUTTER_SEARCH_MIN_RATIO
    high = page_width * GUTTER_SEARCH_MAX_RATIO

    columnar = [
        i
        for i, ln in enumerate(lines)
        if (start := _gutter_start(ln, low, high)) is not None
        and abs(start - approx) <= GUTTER_MIN_WIDTH
    ]
    if len(columnar) < MIN_COLUMN_LINES:
        return lines

    start_idx, end_idx = min(columnar), max(columnar)

    # Refine: the true boundary is just left of the leftmost right-column
    # word. Taking the most common start alone would strand a hanging
    # indent - Fredericton sets the "(b)" marker of Conditional Uses about
    # 14pt left of the text it labels, so a boundary fixed on the text
    # would file "(b)" under Permitted Uses.
    right_starts = [
        s
        for i in columnar
        if (s := _gutter_start(lines[i], low, high)) is not None
    ]
    boundary = min(right_starts) - COLUMN_X_TOLERANCE / 2

    # The list's left column usually runs on past the last row that had a
    # right-column entry. Absorb those trailing lines, but stop at a
    # heading or at any line that straddles the gutter (i.e. prose).
    while end_idx + 1 < len(lines):
        nxt = lines[end_idx + 1]
        text = nxt.text.strip()
        if nxt.starts_bold and (scheme.clause.match(text) or scheme.subsection.match(text)):
            break
        if not _entirely_one_side(nxt, boundary):
            break
        end_idx += 1

    before, region, after = lines[:start_idx], lines[start_idx : end_idx + 1], lines[end_idx + 1 :]

    left_lines: list[_Line] = []
    right_lines: list[_Line] = []
    for line in region:
        left = [w for w in line.words if w["x0"] < boundary]
        right = [w for w in line.words if w["x0"] >= boundary]
        if left:
            left_lines.append(_Line(line.top, left))
        if right:
            right_lines.append(_Line(line.top, right))

    # A genuine second column is a list in its own right, so its lines
    # carry their own markers ("(b) Conditional Uses", "(1) Kennel").
    #
    # Zone standards look superficially identical - a label at the left
    # margin and a value far to the right, separated by more whitespace
    # than any gutter - but the right side is the VALUE of the row it sits
    # on: "(i) Interior Lot:" ... "11.5 metres". Reordering those into two
    # blocks strands every measurement from the rule it belongs to, so a
    # question about minimum lot area retrieves a label with no number and
    # a loose list of numbers that could be paired with anything. Setbacks
    # and lot areas are the most-asked questions this corpus answers, so
    # the check below is what keeps them attached.
    marked = sum(1 for line in right_lines if LIST_MARKER.match(line.text.strip()))
    if marked < MIN_COLUMN_LINES:
        return lines

    return before + left_lines + right_lines + after


# ---------------------------------------------------------------------
#  Page extraction
# ---------------------------------------------------------------------


def _centre(word: dict) -> float:
    return (word["x0"] + word["x1"]) / 2


@dataclass
class _Matrix:
    columns: _Line
    source: _Line
    rows: list[tuple[_Line, str | None]]
    consumed: list[_Line]


def _find_matrix(lines: list[_Line], scheme: NumberingScheme) -> _Matrix | None:
    """Locate a zone-permission matrix: a zone-code header plus symbol rows.

    Fredericton's sign regulations are a grid - rows are sign types, columns
    are the fifteen commercial zones, cells are P/SD/DA. Read line by line
    it flattens to "6.4(1) P P P P ...", which drops the zone mapping
    entirely while still looking like a citable clause. That is the worst
    possible failure for this application: it would let the assistant say a
    sign is permitted somewhere without any basis for the claim.
    """
    header: _Line | None = None
    header_source: _Line | None = None
    rows: list[tuple[_Line, str | None]] = []
    consumed: list[_Line] = []
    category: str | None = None

    for line in lines:
        words = line.words
        if header is None:
            # The column headers are the trailing run of zone codes. The
            # left of that same line carries wrapped legend text ("...
            # Development Agreement LC NC DC ..."), so requiring every
            # word to be a zone code would miss the header entirely.
            run: list[dict] = []
            for word in reversed(words):
                if scheme.zone_code.match(word["text"]):
                    run.append(word)
                else:
                    break
            if len(run) >= MIN_MATRIX_COLUMNS:
                run.reverse()
                header = _Line(line.top, run)
                header_source = line
            continue

        if not words:
            continue

        label, cells = words[0], words[1:]
        if not scheme.matrix_row.match(label["text"]):
            # An all-caps line between rows is the sign-type banner for the
            # rows beneath it ("CANOPY", "SANDWICH BOARD"). Captured so the
            # rendered rows say what they are about, and consumed so it
            # does not drift into a neighbouring clause as loose text.
            text = line.text.strip()
            if text.isupper() and len(text) > 2:
                category = text
                consumed.append(line)
            continue

        # Zero cells is itself a fact: that row has no entry in any listed
        # zone. Accepted here so the row is consumed rather than falling
        # through to be parsed as a clause whose "title" is the next
        # banner line.
        if all(c["text"] in MATRIX_CELL_TOKENS for c in cells):
            rows.append((line, category))

    if header is None or not rows:
        return None
    return _Matrix(columns=header, source=header_source, rows=rows, consumed=consumed)


def _render_matrix(matrix: _Matrix, legend: dict[str, str]) -> list[str]:
    """Render matrix rows as sentences that name the zone for every cell."""
    columns = [(_centre(w), w["text"]) for w in matrix.columns.words]
    rows = matrix.rows
    rendered: list[str] = []

    if legend:
        key = "; ".join(f"{sym} = {meaning}" for sym, meaning in legend.items())
        rendered.append(f"Legend: {key}.")

    for row, category in rows:
        label = row.words[0]["text"]
        prefix = f"{category} {label}" if category else label
        by_symbol: dict[str, list[str]] = collections.defaultdict(list)

        for cell in row.words[1:]:
            centre = _centre(cell)
            nearest = min(columns, key=lambda col: abs(col[0] - centre))
            if abs(nearest[0] - centre) <= MATRIX_COLUMN_TOLERANCE:
                by_symbol[cell["text"]].append(nearest[1])

        if not by_symbol:
            # Stated as an absence of table entries rather than as a
            # prohibition. Reading a blank cell as "not permitted" is an
            # interpretation of the bylaw, and this parser does not make
            # interpretations - it reports what the document contains.
            rendered.append(f"{prefix} - no entry for any listed zone.")
            continue

        parts = [
            f"{legend.get(symbol, symbol)}: {', '.join(zones)}"
            for symbol, zones in by_symbol.items()
        ]
        rendered.append(f"{prefix} - " + "; ".join(parts) + ".")

    return rendered


@dataclass
class PageLine:
    text: str
    starts_bold: bool


@dataclass
class ParsedPage:
    page_number: int
    lines: list[PageLine]
    page_label: str | None
    part_number: str | None
    part_title: str | None
    amendments: list[str]
    matrix_number: str | None = None
    matrix_title: str | None = None
    matrix_lines: list[str] = field(default_factory=list)


def parse_page(page, scheme: NumberingScheme) -> ParsedPage:
    """Extract one page into ordered, header-stripped, column-aware lines."""
    lines = _group_lines(page.extract_words(extra_attrs=["fontname", "size"]))

    part_number: str | None = None
    part_title: str | None = None
    page_label: str | None = None

    # Running header is the topmost line when it matches the header shape.
    if lines:
        header = scheme.running_header.match(lines[0].text.strip())
        if header:
            part_number, part_title = header.group(1), header.group(2).strip()
            lines = lines[1:]

    # Printed page label is the bottom line.
    if lines:
        label = scheme.page_label.match(lines[-1].text.strip())
        if label:
            page_label = lines[-1].text.strip()
            lines = lines[:-1]

    lines, amendments = _strip_amendments(lines, scheme)

    # Matrix detection must precede column linearisation. A sparsely
    # filled matrix row ("6.4(2)(a) P ... SD") contains gutter-sized gaps
    # that would otherwise be mistaken for a column break and reshuffled.
    matrix_number = matrix_title = None
    matrix_lines: list[str] = []
    matrix = _find_matrix(lines, scheme)

    if matrix is not None:
        legend = {}
        for line in lines:
            found = scheme.legend.match(line.text.strip())
            if found:
                legend[found.group(1)] = found.group(2).strip()

        matrix_lines = _render_matrix(matrix, legend)
        matrix_number = _common_clause_prefix(
            [row.words[0]["text"] for row, _ in matrix.rows]
        )

        consumed = {
            id(matrix.source),
            *(id(row) for row, _ in matrix.rows),
            *(id(line) for line in matrix.consumed),
        }
        remaining = [ln for ln in lines if id(ln) not in consumed]

        # The page's banner ("COMMERCIAL ZONES") titles the matrix.
        for line in remaining:
            text = line.text.strip()
            if text.isupper() and not scheme.legend.match(text) and len(text) > 3:
                matrix_title = text
                break
        lines = remaining
    else:
        lines = _linearise_columns(lines, page.width, scheme)

    return ParsedPage(
        page_number=page.page_number,
        lines=[PageLine(ln.text, ln.starts_bold) for ln in lines],
        page_label=page_label,
        part_number=part_number,
        part_title=part_title,
        amendments=amendments,
        matrix_number=matrix_number,
        matrix_title=matrix_title,
        matrix_lines=matrix_lines,
    )


def _common_clause_prefix(labels: list[str]) -> str | None:
    """The shared "X.Y" stem of a set of matrix row labels."""
    stems = {label.split("(")[0] for label in labels}
    return stems.pop() if len(stems) == 1 else None


# ---------------------------------------------------------------------
#  Document assembly
# ---------------------------------------------------------------------


def _looks_like_toc(page: ParsedPage) -> bool:
    """Table-of-contents pages repeat every heading and must not be chunked.

    They are dense in trailing page references ("2-1"), which body prose
    never is.
    """
    if not page.lines:
        return False
    refs = sum(1 for ln in page.lines if re.search(r"\s\d+\s*[-–]\s*\d+$", ln.text))
    return refs >= max(3, len(page.lines) // 3)


def parse_pdf(
    path: str | Path,
    *,
    scheme: NumberingScheme | None = None,
    first_page: int = 1,
    last_page: int | None = None,
) -> list[Clause]:
    """Parse a bylaw PDF into ordered clauses.

    Text before the first recognised clause heading (title page, adoption
    notes) is dropped rather than attached to clause 1: unattributable text
    cannot be cited, and Section 5 requires every statement to carry a
    section reference.
    """
    scheme = scheme or NumberingScheme()
    clauses: list[Clause] = []

    current: Clause | None = None
    buffer: list[str] = []
    parent_number: str | None = None
    parent_title: str | None = None
    part_number: str | None = None
    part_title: str | None = None
    skipped_toc = 0

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = _clean_text("\n".join(buffer))
            if not current.is_empty:
                clauses.append(current)
        current, buffer = None, []

    with pdfplumber.open(str(path)) as pdf:
        pages = pdf.pages[first_page - 1 : last_page]
        for raw_page in pages:
            page = parse_page(raw_page, scheme)

            if _looks_like_toc(page):
                skipped_toc += 1
                continue

            # Some pages carry no running header at all - the sign
            # permission matrices start straight in on "COMMERCIAL ZONES".
            # The document is sequential, so the last header seen still
            # applies; carrying it forward attributes those pages to the
            # right section instead of leaving them unattributed.
            if page.part_number:
                if page.part_number != part_number:
                    # A new part invalidates the subsection carried over
                    # from the previous one, which would otherwise attach
                    # Section 6 content to a Section 5 parent.
                    parent_number = parent_title = None
                part_number, part_title = page.part_number, page.part_title

            if page.matrix_lines:
                flush()
                matrix_part = page.matrix_number.split(".")[0] if page.matrix_number else None
                clauses.append(
                    Clause(
                        section_number=page.matrix_number or (part_number or "?"),
                        section_title=page.matrix_title or "Permissions by zone",
                        text=_clean_text("\n".join(page.matrix_lines)),
                        page_number=page.page_number,
                        page_label=page.page_label,
                        part_number=matrix_part or part_number,
                        part_title=page.part_title or part_title,
                        # A matrix is a standalone table; its own number is
                        # the parent. Inheriting the subsection last seen
                        # would file it under an unrelated section.
                        parent_number=None,
                        parent_title=None,
                        amendments=list(page.amendments),
                    )
                )

            in_definitions = scheme.is_definitions_part(part_title)

            for line in page.lines:
                stripped = line.text.strip()
                if not stripped:
                    continue

                # Section 3 numbers its entries "(203) Utilities means ..."
                # rather than "X.Y(N)". Without this branch the whole
                # definitions chapter accumulates onto the last clause of
                # Section 2 - one 58,000-character blob that exceeds the
                # embedding input limit and cites the wrong section for
                # every term in it.
                if in_definitions:
                    definition = scheme.definition.match(stripped)
                    if definition:
                        flush()
                        number, body = definition.group(1), definition.group(2).strip()
                        current = Clause(
                            section_number=f"{part_number}({number})",
                            section_title=_definition_term(body),
                            text="",
                            page_number=page.page_number,
                            page_label=page.page_label,
                            part_number=part_number,
                            part_title=part_title,
                            parent_number=part_number,
                            parent_title=part_title,
                            amendments=list(page.amendments),
                        )
                        buffer.append(body)
                        continue

                clause_match = (
                    scheme.clause.match(stripped) if line.starts_bold else None
                )
                if clause_match:
                    flush()
                    number, title = clause_match.group(1), clause_match.group(2).strip()
                    current = Clause(
                        section_number=number,
                        section_title=title,
                        text="",
                        page_number=page.page_number,
                        page_label=page.page_label,
                        part_number=part_number,
                        part_title=part_title,
                        parent_number=parent_number,
                        parent_title=parent_title,
                        amendments=list(page.amendments),
                    )
                    continue

                subsection_match = (
                    scheme.subsection.match(stripped) if line.starts_bold else None
                )
                if subsection_match:
                    flush()
                    parent_number = subsection_match.group(1)
                    parent_title = subsection_match.group(2).strip()
                    continue

                if current is not None:
                    buffer.append(stripped)

        flush()

    log.info(
        "pdf_parsed",
        path=str(path),
        clauses=len(clauses),
        toc_pages_skipped=skipped_toc,
    )
    return clauses


def _definition_term(body: str) -> str:
    """The defined term at the head of a definition entry.

    English uses "X means ...":
        "Convenience Store means a use not exceeding 300 square metres"
        -> "Convenience Store"

    French wraps the term in guillemets instead:
        "« abri d'auto » Garage privé ..."
        -> "abri d'auto"

    The section title is weighted above body text in the full-text index
    (see content_tsv in 001_init_schema.sql), so a title truncated to a
    word count would put "Établissement où l'on" in the highest-weighted
    field and bury the actual term being defined.
    """
    quoted = re.match(r"^[«\"“]\s*(.{1,80}?)\s*[»\"”]", body)
    if quoted:
        return quoted.group(1).strip()

    match = re.match(r"^(.{1,80}?)\s+means\b", body)
    if match:
        return match.group(1).strip()

    return " ".join(body.split()[:6])


def _clean_text(text: str) -> str:
    """Normalise whitespace without altering wording or punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    # Sentence-final punctuation is often its own positioned glyph, which
    # the word join turns into "minimum lot area ." Closing the gap is
    # presentation only - no character is added or removed.
    text = re.sub(r" +([.,;:)])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_hash(path: str | Path) -> str:
    """SHA-256 of the source file, for the Step 4 change detection."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
