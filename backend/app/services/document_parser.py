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

# A glyph shorter than this fraction of the preceding one, AND sitting on a
# raised baseline, is a superscript.
SUPERSCRIPT_SIZE_RATIO = 0.8

# How far above the preceding word's baseline the glyph must sit, as a
# fraction of that word's height. Height alone is not enough: a digit is
# already shorter than a word with ascenders, so "case 2" would qualify.
# A raised baseline is what actually distinguishes m² from m 2.
SUPERSCRIPT_RISE_RATIO = 0.15

SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _height(word: dict) -> float:
    return float(word.get("bottom", 0.0)) - float(word.get("top", 0.0))


def _is_superscript(word: dict, previous: dict) -> bool:
    """Whether `word` is set as a superscript of the word before it.

    Measured from the word's own box rather than from a `size` attribute.
    Requesting `size` in extract_words() makes pdfplumber start a new word
    wherever font size changes, which silently re-segments the whole
    document: the zone codes in the sign matrix came apart into "L", "O",
    "C", "M" instead of LC, OC, COR-1, MX-1, and were ingested that way.
    """
    height, prior = _height(word), _height(previous)
    if height <= 0 or prior <= 0:
        return False

    smaller = height < prior * SUPERSCRIPT_SIZE_RATIO
    raised = float(word.get("bottom", 0.0)) < (
        float(previous.get("bottom", 0.0)) - prior * SUPERSCRIPT_RISE_RATIO
    )
    return smaller and raised


SCHEME_HEALTH_MAX_CHARS = 12_000


def scheme_health(clauses: list["Clause"]) -> dict:
    """Summary used to judge whether a numbering scheme actually fits.

    Clause count alone is misleading. A scheme that matches only some
    headings still produces clauses - each one swallowing everything up to
    the next match - so a 100,000-character "clause" means the scheme is
    wrong even though the parse "succeeded". Fredericton's definitions
    chapter failed exactly this way at 58,000 characters.
    """
    if not clauses:
        return {"clauses": 0, "median": 0, "max": 0, "oversized": 0, "healthy": False}

    lengths = sorted(len(c.text) for c in clauses)
    oversized = sum(1 for n in lengths if n > SCHEME_HEALTH_MAX_CHARS)
    return {
        "clauses": len(clauses),
        "median": lengths[len(lengths) // 2],
        "max": lengths[-1],
        "oversized": oversized,
        "healthy": oversized == 0 and len(clauses) > 20,
    }


def _strip_accents(text: str) -> str:
    """Fold accented characters to their base letters, for matching only.

    Never applied to stored bylaw text - only to headings being classified,
    so "Définitions" and "Definitions" take the same branch.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Below this many columns it is not a matrix, just a short line.
MIN_MATRIX_COLUMNS = 5


@dataclass(frozen=True)
class NumberingScheme:
    """Regexes describing one municipality's clause numbering.

    Fredericton's conventions are the defaults. Other municipalities pick
    a named variant from SCHEMES below rather than editing these, because
    a change here silently re-parses every already-verified corpus.
    """

    # "8.14(4) Standards" — the citable clause level.
    clause: re.Pattern[str] = re.compile(r"^(\d+\.\d+\(\d+[a-z]?\))\s*(.*)$")
    # "2.1 OPERATION" — the subsection heading above clauses.
    subsection: re.Pattern[str] = re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Z0-9 &/,'’()\-\.]{3,})$")
    # "(203) Utilities means ..." — numbered definition entries (Z-5).
    definition: re.Pattern[str] = re.compile(r"^\((\d+)\)\s+(.+)$")

    # Unnumbered definitions: `<term> means <text>`. Every bylaw in the
    # pilot set outside Fredericton defines terms this way and numbers
    # none of them - Saint John writes '"block" means ...', Mount Pearl
    # '"LOT FRONTAGE" means ...', St. John's 'PLACE OF ASSEMBLY means ...'.
    # Without this, a definitions chapter matches nothing and accumulates
    # onto the previous clause: Saint John produced a single 105,000-
    # character "clause" that way, far past the embedding input limit and
    # mis-citing every term inside it.
    definition_by_means: re.Pattern[str] = re.compile(
        "^[\"\u201c\u2018']?"                    # optional opening quote
        "([A-Za-z][^\"\u201c\u201d\u2018\u2019]{1,70}?)"  # the defined term
        "[\"\u201d\u2019']?"                     # optional closing quote
        r"\s+means\b"                        # the defining verb
    )

    # Quoted definitions, for bylaws that do not set their terms in
    # bold. Moncton writes: “ bicycle parking space ” means a slot ...
    # The opening quote is what makes this safe to match without the
    # bold requirement - it is a deliberate typographic marker, whereas
    # an unquoted line containing "means" is usually prose. Continuation
    # lines such as "...secured by means of an 8 inch U lock" match the
    # unquoted pattern and would otherwise open a bogus definition.
    definition_quoted: re.Pattern[str] = re.compile(
        "^[\u201c\u201d\"\u2018\u2019]\\s*"
        # A full stop bars a quoted sentence ("No parking.") from being
        # read as a defined term; no term in these bylaws contains one.
        "([^.“”\"‘’]{1,70}?)"
        "\\s*[\u201c\u201d\"\u2018\u2019]"
        # A qualifier may sit between the term and the verb:
        # "city", when used alone, means ... It is bounded, and barred
        # from containing a full stop or a quote so it can neither run
        # into the next sentence nor swallow a following quoted term.
        r"[^.\u201c\u201d\"]{0,44}?means\b"
    )

    # French definitions, which carry no defining verb at all:
    # « arbre de rue » Arbre à planter entre la limite du lot ...
    # The guillemet opening the line is the entire marker, so the
    # pattern also demands a capitalised body: a cross-reference reads
    # « ... » de l'arrêté and must not be mistaken for a definition.
    definition_guillemet: re.Pattern[str] = re.compile(
        r"^\u00ab\s*"
        r"([^.\u00ab\u00bb]{1,90}?)"
        r"\s*\u00bb\s+"
        "(?=[A-Z\u00c0-\u00d6\u00d8-\u00de0-9])"
    )
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
    #: A zone code column heading. The leading hyphen is Fredericton's
    #: overlay column ("-H"): it sits at the end of the sign matrix
    #: header, and because the header is found by scanning back from the
    #: end of the line, one unrecognised token there hid the whole
    #: heading and a row of "P"s was used instead.
    zone_code: re.Pattern[str] = re.compile(r"^-?[A-Z]{1,5}(?:-\d+)?$")
    # A row label in a permission matrix: "6.4(2)(a)".
    matrix_row: re.Pattern[str] = re.compile(r"^\d+\.\d+(?:\([0-9a-zA-Z]+\))+$")
    # Legend entry: "P = Permitted". Scanned rather than matched, because
    # CBRM sets its whole legend on one line - "P = Permitted as-of-right
    # C = Permitted with additional conditions SP = Site Plan Approval" -
    # and anchoring on the first symbol made the meaning of P the entire
    # rest of the line, which then appeared in every rendered row.
    legend: re.Pattern[str] = re.compile(
        r"\b([A-Z]{1,3})\s*=\s*(.+?)(?=\s+[A-Z]{1,3}\s*=|$)"
    )

    # Whether a matrix row may be labelled by name rather than by clause
    # number. CBRM's use tables are labelled "Dwelling, Two Unit"; opting
    # in per document rather than accepting both everywhere, because on a
    # numbered matrix any stray line ending in a symbol then reads as a
    # row and corrupts the number the whole matrix is cited under.
    matrix_named_rows: bool = False

    # How many columns a zone header needs. CBRM's commercial table has
    # three, and demanding five made it unrecognisable.
    matrix_min_columns: int = MIN_MATRIX_COLUMNS

    # Whether a bracketed clause number is qualified by its parent. In
    # St. John's zone chapters "(1) PERMITTED USES" appears under every
    # zone, so the number alone names 43 different provisions.
    clause_number_takes_parent: bool = False

    # Whether a numbered heading carries a title. Fredericton writes
    # "8.14(4) Fences"; Moncton's numbered provisions are paragraphs of
    # running text with no heading at all, so capturing their first line
    # as a title both truncates it into a fragment and makes the citation
    # read as a heading that the bylaw never wrote.
    clause_titles: bool = True

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
MATRIX_CELL_TOKENS = {"P", "SD", "DA", "SP", "C", "X", "A", "-", "•"}

# Column centres closer than this belong to the same matrix column.
MATRIX_COLUMN_TOLERANCE = 14.0

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
class _MatrixRow:
    """One row of a permission matrix, split into its label and its cells.

    The label is however many words precede the run of symbols: a clause
    number in Fredericton ("6.4(2)(a) P P SD"), the name of the use in
    CBRM ("Dwelling, Two Unit P P P P P P").
    """

    line: _Line
    category: str | None
    label_words: list[dict]
    cells: list[dict]

    @property
    def label(self) -> str:
        return " ".join(w["text"] for w in self.label_words)


@dataclass
class _Matrix:
    columns: _Line
    source: _Line
    rows: list[_MatrixRow]
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
            # A run of identical symbols is a row of cells, not a
            # heading. Without this a table with fewer zone columns than
            # the minimum skips its real header and takes the first row
            # of "P"s instead, and every mark is then reported against a
            # zone named "P".
            if all(w["text"] in MATRIX_CELL_TOKENS for w in run):
                continue

            if len(run) >= scheme.matrix_min_columns:
                run.reverse()
                header = _Line(line.top, run)
                header_source = line
            continue

        if not words:
            continue

        # The cells are the trailing run of symbols; whatever precedes
        # them is the label. Requiring a single-word label would miss
        # every row of a table whose rows are named rather than numbered,
        # and those rows then flatten to "Dwelling, Two Unit P P P P P P"
        # - which silently shifts each mark one zone to the left wherever
        # a cell is blank.
        split = len(words)
        while split > 0 and words[split - 1]["text"] in MATRIX_CELL_TOKENS:
            split -= 1
        label_words, cells = words[:split], words[split:]

        # A numbered label stands as a row even with no cells at all:
        # "6.4(2)(a)" alone means that provision has no entry in any
        # listed zone, which is a fact worth rendering rather than a line
        # to drop. A named label needs at least one cell, because without
        # one there is nothing to distinguish it from a banner.
        numbered = (
            len(label_words) == 1
            and scheme.matrix_row.match(label_words[0]["text"]) is not None
        )
        named = scheme.matrix_named_rows and bool(cells)
        if not label_words or not (numbered or named):
            # An all-caps line between rows is the banner for the rows
            # beneath it ("CANOPY", "SANDWICH BOARD"). Captured so the
            # rendered rows say what they are about, and consumed so it
            # does not drift into a neighbouring clause as loose text.
            text = line.text.strip()
            if text.isupper() and len(text) > 2:
                category = text
                consumed.append(line)
            continue

        rows.append(_MatrixRow(line, category, label_words, cells))

    if header is None or not rows:
        return None
    return _Matrix(columns=header, source=header_source, rows=rows, consumed=consumed)


#: A banner sharing the legend's line: "P = Permitted LIMITED DEVELOPMENT
#: ZONES", where the last three words head the columns rather than
#: explain the symbol.
LEGEND_BANNER_TAIL = re.compile(r"(?:\s+[A-Z][A-Z-]{1,}){2,}$")


def _legend_meaning(text: str) -> str:
    """What a legend symbol means, without the banner beside it.

    Left in, the banner is repeated in every rendered row of the matrix:
    "CANOPY 6.4(1) - Permitted LIMITED DEVELOPMENT ZONES: I-2, IEX ...".
    """
    return LEGEND_BANNER_TAIL.sub("", text.strip()).strip()


def _render_matrix(matrix: _Matrix, legend: dict[str, str]) -> list[str]:
    """Render matrix rows as sentences that name the zone for every cell."""
    columns = [(_centre(w), w["text"]) for w in matrix.columns.words]
    rows = matrix.rows
    rendered: list[str] = []

    if legend:
        key = "; ".join(f"{sym} = {meaning}" for sym, meaning in legend.items())
        rendered.append(f"Legend: {key}.")

    for row in rows:
        prefix = f"{row.category} {row.label}" if row.category else row.label
        by_symbol: dict[str, list[str]] = collections.defaultdict(list)

        for cell in row.cells:
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

    # Set for a page that cannot be attributed to the surrounding clause
    # flow - a full-width table in a bilingual document. Without this the
    # page is appended to whatever clause was open, which both inflates
    # that clause and files the table under an unrelated section number.
    standalone: bool = False


def parse_page(
    page,
    scheme: NumberingScheme,
    *,
    words: list[dict] | None = None,
    page_width: float | None = None,
) -> ParsedPage:
    """Extract one page into ordered, header-stripped, column-aware lines.

    `words` lets a caller supply a subset - one half of a bilingual
    page, say - so the same extraction runs over it unchanged.
    """
    # Only fontname: adding "size" re-segments words wherever the size
    # changes mid-word, which corrupted the sign-matrix zone codes.
    if words is None:
        words = page.extract_words(extra_attrs=["fontname"])
    lines = _group_lines(words)

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
    width = page_width or page.width
    matrix = _find_matrix(lines, scheme)

    if matrix is not None:
        legend = {}
        for line in lines:
            for found in scheme.legend.finditer(line.text.strip()):
                legend[found.group(1)] = _legend_meaning(found.group(2))

        matrix_lines = _render_matrix(matrix, legend)

        # Numbered rows share a stem - "6.4(1)", "6.4(2)" - and that stem
        # is what the matrix is cited under. Named rows have none, so the
        # number comes from the page's own heading instead; without it the
        # table is cited as "?" and a reader cannot look it up.
        matrix_number = _common_clause_prefix(
            [row.label_words[0]["text"] for row in matrix.rows]
        )
        if matrix_number is None:
            for line in lines:
                heading = scheme.clause.match(line.text.strip())
                if heading:
                    matrix_number = heading.group(1)
                    break

        consumed = {
            id(matrix.source),
            *(id(row.line) for row in matrix.rows),
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
        lines = _linearise_columns(lines, page_width or page.width, scheme)

    # A schedule page belongs to no section. Left in the clause flow, its
    # caption and its rotated margin stamps append to whichever section
    # came last - Saint John's final section absorbed thirty such pages.
    standalone = bool(lines) and bool(
        SCHEDULE_CAPTION.match(lines[0].text.strip())
    )

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
        standalone=standalone,
    )


def _common_clause_prefix(labels: list[str]) -> str | None:
    """The shared "X.Y" stem of a set of matrix row labels."""
    stems = {label.split("(")[0] for label in labels}
    return stems.pop() if len(stems) == 1 else None


# ---------------------------------------------------------------------
#  Document assembly
# ---------------------------------------------------------------------


#: "Schedule K: Spruce Lake Industrial (SLI) Zone Setbacks". A schedule
#: is a map or chart with a caption, and it is cited by that caption.
SCHEDULE_CAPTION = re.compile(
    r"^Schedule\s+([A-Z]{1,2})\s*[:.\-\u2013]\s*(.+)$", re.IGNORECASE
)

TABLE_CAPTION = re.compile(r"^(?:TABLE|TABLEAU)\s+(\d+(?:\.\d+)*)\b(.*)$", re.IGNORECASE)


def _table_identity(page: ParsedPage) -> tuple[str | None, str | None]:
    """Number and title for a standalone page, from its caption.

    A caption is what makes the page citable. "TABLE 12.1" is how the
    surrounding clauses refer to it ("listed in Table 12.1"), so citing it
    that way lets a reader follow the reference. Schedules are referred to
    the same way ("as delineated by Schedule C").
    """
    for line in page.lines[:12]:
        text = line.text.strip()
        found = TABLE_CAPTION.match(text)
        if found:
            title = found.group(2).strip(" -\u2013")
            return f"Table {found.group(1)}", title or None

        found = SCHEDULE_CAPTION.match(text)
        if found:
            return f"Schedule {found.group(1).upper()}", found.group(2).strip()
    return None, None


#: A trailing page reference, in the two forms bylaws use: Fredericton's
#: "2-1", and Saint John's dot leader running out to "247".
TOC_PAGE_REFERENCE = re.compile(r"\s\d+\s*[-–]\s*\d+$")
TOC_DOT_LEADER = re.compile(r"\.{4,}\s*\d+\s*$")

#: A tab-index entry: a part name with its number, "Residential Zones 10".
TAB_INDEX_ENTRY = re.compile(r"^[A-Z][A-Za-z ,:&/-]{3,60}\s+\d{1,2}$")
TAB_INDEX_MIN_LINES = 8
TAB_INDEX_RATIO = 0.6


def _looks_like_toc(page: ParsedPage) -> bool:
    """Table-of-contents pages repeat every heading and must not be chunked.

    They are dense in trailing page references, which body prose never is.
    """
    if not page.lines:
        return False
    refs = sum(
        1
        for ln in page.lines
        if TOC_PAGE_REFERENCE.search(ln.text) or TOC_DOT_LEADER.search(ln.text)
    )
    return refs >= max(3, len(page.lines) // 3)


def _looks_like_divider(page: ParsedPage) -> bool:
    """A part divider carrying only the document's tab index.

    Saint John opens each part with a page listing every part and its
    number, and nothing else. The list states no rule, but it does name
    fourteen parts, so leaving it in appends that list to whichever
    clause was open and puts "Residential Zones 10" inside a provision
    about parking.

    Measured at 0.76-0.81 on those pages against 0.03 on any other.
    """
    lines = [ln.text.strip() for ln in page.lines if ln.text.strip()]
    if len(lines) < TAB_INDEX_MIN_LINES:
        return False
    entries = sum(1 for text in lines if TAB_INDEX_ENTRY.match(text))
    return entries / len(lines) >= TAB_INDEX_RATIO



#: A contents entry: a numbered heading, or a Division/Part line, short
#: enough to be a heading rather than a provision.
CONTENTS_ENTRY = re.compile(
    r"^(?:\d{1,3}(?:\.\d+)?(?:\s?\(\d+\))?\s+\S"
    r"|Division\s+\d|Section\s+\d|PART(?:IE)?\s+\d|SCHEDULE\s|ANNEXE\s)",
    re.IGNORECASE,
)
CONTENTS_ENTRY_MAX_CHARS = 70
CONTENTS_MIN_LINES = 12
CONTENTS_ENTRY_RATIO = 0.5


def _looks_like_contents(page: ParsedPage) -> bool:
    """A contents listing with no page references, which Moncton's is.

    `_looks_like_toc` keys on trailing page numbers ("2-1"); Moncton's
    contents pages carry none, so they read as a run of valid clause
    headings and would duplicate every section in the bylaw as an empty
    clause. What marks them instead is that almost every line is a short
    numbered heading - measured at 0.66-0.92 on Moncton's contents pages
    against at most 0.11 on any body page.
    """
    lines = [ln for ln in page.lines if ln.text.strip()]
    if len(lines) < CONTENTS_MIN_LINES:
        return False

    entries = sum(
        1
        for ln in lines
        if len(ln.text.strip()) < CONTENTS_ENTRY_MAX_CHARS
        and CONTENTS_ENTRY.match(ln.text.strip())
    )
    return entries / len(lines) >= CONTENTS_ENTRY_RATIO


#: One entry of a bilingual term index: "accessory building – bâtiment
#: accessoire". No sentence punctuation, and lowercase on both sides.
TERM_INDEX_ENTRY = re.compile(
    r"^[a-z\u00e0-\u00ff][^\u2013\u2014.]{1,44}[\u2013\u2014][^\u2013\u2014.]{1,44}$"
)
TERM_INDEX_RATIO = 0.4


def _looks_like_term_index(page: ParsedPage) -> bool:
    """The English-French index of defined terms, which is not provisions.

    Moncton opens with five pages pairing each defined term with its
    translation. They sit under the heading "1 Definitions", so parsing
    them produces a 10,000-character clause cited as section 1 - the same
    citation as the real section 1, which is on a later page. A reader
    following that citation would land on a word list.

    Measured at 0.58-0.81 on those five pages against at most 0.07 on any
    page of provisions.
    """
    lines = [ln.text.strip() for ln in page.lines if ln.text.strip()]
    if len(lines) < CONTENTS_MIN_LINES:
        return False
    entries = sum(1 for text in lines if TERM_INDEX_ENTRY.match(text))
    return entries / len(lines) >= TERM_INDEX_RATIO


#: The header of an amendment register, repeated on every page of one.
REGISTER_DATE_HEADER = re.compile(r"\bPublished\s+Date\b", re.IGNORECASE)
REGISTER_AMENDMENT_HEADER = re.compile(r"\bAmendment\b", re.IGNORECASE)
REGISTER_HEADER_LINES = 8


def _looks_like_amendment_register(page: ParsedPage) -> bool:
    """A log of past amendments rather than the regulations themselves.

    Mount Pearl appends twenty-five pages recording each amendment and
    the text it inserted. Every one of those insertions already appears
    in the consolidated body - "INDOOR PARKING FACILITIES" is defined on
    page 10 and recorded again on page 179 - so parsing the register
    duplicates provisions and, because its entries carry no section
    number of their own, files them under whatever section came last. Two
    clauses of 14,030 and 13,304 characters were built that way.

    Recognised by the table header the register repeats on every page,
    which the front matter does not carry even where it names an
    amendment.
    """
    top = [ln.text for ln in page.lines[:REGISTER_HEADER_LINES]]
    return any(REGISTER_DATE_HEADER.search(t) for t in top) and any(
        REGISTER_AMENDMENT_HEADER.search(t) for t in top
    )

def assemble_clauses(
    pages: list[ParsedPage],
    scheme: NumberingScheme,
) -> list[Clause]:
    """Turn parsed pages into ordered clauses.

    Separated from PDF reading so the same assembly runs over a whole
    document or over one language's half of a bilingual one.

    Text before the first recognised clause heading (title page, adoption
    notes) is dropped rather than attached to clause 1: unattributable
    text cannot be cited, and Section 5 requires every statement to carry
    a section reference.
    """
    clauses: list[Clause] = []

    current: Clause | None = None
    buffer: list[str] = []
    parent_number: str | None = None
    parent_title: str | None = None
    part_number: str | None = None
    part_title: str | None = None
    definition_index = 0
    skipped_toc = 0
    # A use table running over several pages carries its heading only on
    # the first. The continuation pages are the same table, so they are
    # cited under the same number rather than as "?".
    last_matrix_number: str | None = None
    # The section an unnumbered definition sits inside. Moncton's and
    # Saint John's definitions are alphabetical and carry no number of
    # their own, so a number invented for them would cite a provision
    # the bylaw does not have. They are cited under their section, with
    # the defined term as the heading, which is how a reader finds one.
    enclosing_number: str | None = None
    # Set by a heading, cleared by the next line that is not a clause. A
    # section number on the line directly beneath a heading is exempt
    # from the bold gate; without that, Summerside loses the first
    # provision of all seventeen of its zone chapters, because the zone
    # heading above it is consumed as the parent rather than held as a
    # pending title.
    after_heading = False
    # Moncton heads a section with its title on the line above the
    # number - "Sight triangle and setback ..." then "111 (1) Minimum
    # yard requirements ...". Held here until a clause claims it.
    pending_title: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = _clean_text("\n".join(buffer))
            if not current.is_empty:
                clauses.append(current)
        current, buffer = None, []

    for page in pages:
        if (
            _looks_like_toc(page)
            or _looks_like_contents(page)
            or _looks_like_term_index(page)
            or _looks_like_divider(page)
            or _looks_like_amendment_register(page)
        ):
            skipped_toc += 1
            continue

        # Some pages carry no running header at all - the sign permission
        # matrices start straight in on "COMMERCIAL ZONES". The document
        # is sequential, so the last header seen still applies.
        if page.part_number:
            if page.part_number != part_number:
                # A new part invalidates the subsection carried over from
                # the previous one.
                parent_number = parent_title = None
            part_number, part_title = page.part_number, page.part_title

        if page.matrix_lines:
            flush()
            if page.matrix_number:
                last_matrix_number = page.matrix_number
            matrix_part = (
                page.matrix_number.split(".")[0] if page.matrix_number else None
            )
            clauses.append(
                Clause(
                    section_number=(
                        page.matrix_number or last_matrix_number or part_number or "?"
                    ),
                    section_title=page.matrix_title or "Permissions by zone",
                    text=_clean_text("\n".join(page.matrix_lines)),
                    page_number=page.page_number,
                    page_label=page.page_label,
                    part_number=matrix_part or part_number,
                    part_title=page.part_title or part_title,
                    # A matrix is a standalone table; its own number is the
                    # parent. Inheriting the last subsection would file it
                    # under an unrelated section.
                    parent_number=None,
                    parent_title=None,
                    amendments=list(page.amendments),
                )
            )

        if page.standalone:
            # Emitted whole rather than merged into the clause flow. Its
            # rows are a grid whose meaning depends on the whole page, and
            # attributing it to the previous clause would cite a setback
            # table as part of an unrelated provision.
            flush()
            text = _clean_text("\n".join(line.text for line in page.lines))
            if text:
                number, title = _table_identity(page)
                clauses.append(
                    Clause(
                        section_number=number
                        or f"p.{page.page_label or page.page_number}",
                        section_title=title or "Table",
                        text=text,
                        page_number=page.page_number,
                        page_label=page.page_label,
                        part_number=part_number,
                        part_title=part_title,
                        parent_number=None,
                        parent_title=None,
                        amendments=list(page.amendments),
                    )
                )
            continue

        in_definitions = scheme.is_definitions_part(part_title)

        # Indexed rather than iterated: a French definition whose term is
        # too long for one line closes its guillemet on the next, and the
        # pattern can only see that if the branch may look ahead.
        index = 0
        while index < len(page.lines):
            line = page.lines[index]
            index += 1
            stripped = line.text.strip()
            if not stripped:
                continue

            # Order matters. A numbered heading wins over anything else on
            # the line, then the subsection heading, and only then the two
            # definition forms - otherwise a heading whose title happens
            # to contain "means" would open a definition, not a clause.
            # The bold gate keeps prose from opening a clause. A line
            # directly under a heading is exempt: the French column writes
            # its section number unbolded and with a trailing period
            # ("1. Sauf indication contraire ..."), so section 1 was never
            # detected and every French definition was cited under a
            # section number that does not exist.
            clause_match = (
                scheme.clause.match(stripped)
                if line.starts_bold
                or pending_title
                # Only where the heading precedes the number. Applied to a
                # scheme whose headings carry their own number, it lets
                # ordinary prose under any heading open a clause: Mount
                # Pearl gained 41 spurious ones and an 18,537-character
                # blob.
                or (after_heading and not scheme.clause_titles)
                else None
            )
            if clause_match:
                flush()
                # "82. 1 (1)" and "111 (1)" are the same citation once
                # the layout's spacing is taken out: 82.1(1), 111(1).
                # A scheme may name its groups when one pattern covers
                # more than one heading shape: St. John's "SECTION 2-
                # DEFINITIONS" carries the section that its unnumbered
                # definitions are cited under, and its zone chapters
                # number their provisions "(1)".
                named = clause_match.groupdict()
                raw_number = (
                    named.get("number") or named.get("part") or clause_match.group(1)
                )
                number = re.sub(r"\s+", "", raw_number)
                if (
                    scheme.clause_number_takes_parent
                    and parent_number
                    and named.get("number")
                ):
                    number = f"{parent_number}{number}"
                rest = (
                    named.get("title") or named.get("parttitle") or clause_match.group(2)
                ).strip()
                # An untitled scheme keeps its first line as text. Read as
                # a title it would be a truncated sentence, and the clause
                # body would begin mid-sentence.
                title = (
                    rest if scheme.clause_titles else " ".join(pending_title)
                )
                pending_title = []
                current = Clause(
                    section_number=number,
                    section_title=title or None,
                    text="",
                    page_number=page.page_number,
                    page_label=page.page_label,
                    part_number=part_number,
                    part_title=part_title,
                    parent_number=parent_number,
                    parent_title=parent_title,
                    amendments=list(page.amendments),
                )
                enclosing_number = number
                after_heading = False
                if not scheme.clause_titles and rest:
                    buffer.append(rest)
                continue

            subsection_match = (
                scheme.subsection.match(stripped) if line.starts_bold else None
            )
            if subsection_match:
                flush()
                # Named groups let a heading put its code after its title
                # ("MINI HOME PARK (MHP) ZONE") without reversing what
                # every other scheme means by group 1 and group 2.
                groups = subsection_match.groupdict()
                parent_number = groups.get("number") or subsection_match.group(1)
                parent_title = (
                    groups.get("title") or subsection_match.group(2)
                ).strip()
                after_heading = True
                continue

            # Fredericton numbers its entries "(203) Utilities means ...".
            # Without this branch the definitions chapter accumulates onto
            # the last clause of the previous section.
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

            # Unnumbered `<term> means ...`, checked regardless of whether
            # the running header names a definitions chapter: only
            # Fredericton labels its header that way.
            #
            # Two forms. A quoted term needs no bold - the quotes are the
            # marker. An unquoted one does, because prose containing
            # "means" is common and is never bold-started.
            by_means = scheme.definition_quoted.match(stripped)
            if by_means is None and line.starts_bold:
                by_means = scheme.definition_by_means.match(stripped)
            if by_means is not None:
                flush()
                definition_index += 1
                current = Clause(
                    section_number=enclosing_number or part_number or "0",
                    section_title=by_means.group(1).strip(),
                    text="",
                    page_number=page.page_number,
                    page_label=page.page_label,
                    part_number=part_number,
                    part_title=part_title,
                    parent_number=part_number,
                    parent_title=part_title,
                    amendments=list(page.amendments),
                )
                buffer.append(stripped)
                continue

            # French definitions: « terme » Corps du texte. No verb to
            # anchor on, so a wrapped term is joined with the line that
            # carries its closing guillemet before matching - otherwise
            # the three longest French terms open no clause and their
            # definitions land in the previous one.
            guillemet = scheme.definition_guillemet.match(stripped)
            if (
                guillemet is None
                and stripped.startswith("\u00ab")
                and "\u00bb" not in stripped
                and index < len(page.lines)
            ):
                joined = f"{stripped} {page.lines[index].text.strip()}"
                guillemet = scheme.definition_guillemet.match(joined)
                if guillemet is not None:
                    stripped = joined
                    index += 1

            if guillemet is not None:
                flush()
                definition_index += 1
                current = Clause(
                    section_number=enclosing_number or part_number or "0",
                    section_title=guillemet.group(1).strip(),
                    text="",
                    page_number=page.page_number,
                    page_label=page.page_label,
                    part_number=part_number,
                    part_title=part_title,
                    parent_number=part_number,
                    parent_title=part_title,
                    amendments=list(page.amendments),
                )
                buffer.append(stripped)
                continue

            if not scheme.clause_titles and line.starts_bold:
                headings = pending_title + [stripped]
                # Two lines and 140 characters is the longest heading in
                # the bylaw. Beyond that it is a bold table header, not a
                # title, and it belongs in the text.
                if len(headings) <= 2 and len(" ".join(headings)) <= 140:
                    pending_title = headings
                    continue

            if pending_title:
                # Bold, but no clause claimed it - a table header rather
                # than a heading. Kept as text rather than discarded.
                if current is not None:
                    buffer.extend(pending_title)
                pending_title = []

            after_heading = False
            if current is not None:
                buffer.append(stripped)

    flush()

    log.info(
        "clauses_assembled", clauses=len(clauses), toc_pages_skipped=skipped_toc
    )
    return clauses


def parse_pdf(
    path: str | Path,
    *,
    scheme: NumberingScheme | None = None,
    first_page: int = 1,
    last_page: int | None = None,
) -> list[Clause]:
    """Parse a bylaw PDF into ordered clauses."""
    scheme = scheme or NumberingScheme()

    with pdfplumber.open(str(path)) as pdf:
        pages = [
            parse_page(raw, scheme)
            for raw in pdf.pages[first_page - 1 : last_page]
        ]

    clauses = assemble_clauses(pages, scheme)
    log.info("pdf_parsed", path=str(path), clauses=len(clauses))
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


# ---------------------------------------------------------------------
#  Named schemes
#
#  Selected per municipality by `numbering` in municipalities_config.json.
#  Measured against the real documents, not assumed: the counts in each
#  comment come from parsing the published PDF.
# ---------------------------------------------------------------------

#: "8.14(4) Standards" - Fredericton Z-5 and Saint John ZoneSJ.
CLAUSE_PAREN = re.compile(r"^(\d+\.\d+\(\d+[a-z]?\))\s*(.*)$")

#: "1.2.1 Words and phrases" - Mount Pearl, CBRM.
CLAUSE_DOTTED3 = re.compile(r"^(\d+\.\d+\.\d+)\s+(.*)$")

#: "1.1 Short Title" - Summerside, St. John's.
CLAUSE_DOTTED2 = re.compile(r"^(\d+\.\d+)\s+(.*)$")

SUBSECTION_DOTTED = re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Za-z0-9 &/,'’()\-\.]{3,})$")
SUBSECTION_NUMBER = re.compile(r"^(\d+)\s+([A-Z][A-Z0-9 &/,'’()\-\.]{3,})$")

#: "100 (4)", "100.1", "108(1)" - Moncton Z-222. Sections are numbered
#: straight through the document rather than per chapter, so there is no
#: leading chapter number to anchor on.
#
# Every part is optional after the first, because a section whose whole
# content is one paragraph carries no subsection number at all ("142 In
# accordance with section 7 ..."). The spacing is optional too:
# pdfplumber emits "82.1" as "82" then ". 1" where the glyph spacing
# changes, and the number is normalised on the way into the clause.
#
# Only ever applied to a bold line, so a table row opening with a figure
# cannot match it.
CLAUSE_MONCTON = re.compile(
    r"^(\d{1,3}(?:\s?\.\s?\d+)?(?:\s?\(\d+\))?)\.?\s+(.*)$"
)

#: "Division 8.2 Other residential uses" / "Section 8.2 Autres usages".
SUBSECTION_DIVISION = re.compile(
    r"^(?:Division|Section)\s+(\d+\.\d+)\s+(.+)$", re.IGNORECASE
)


#: Saint John heads a section either way: "4.2(5) PARKING LOT STANDARDS"
#: or "4.4 Drive-Thru Facilities". Recognising only the first merged
#: everything under a title-case heading into the clause above it - one
#: clause ran from parking standards to signs, thirty pages later.
#:
#: The title must open with a capital, which is what separates a heading
#: from a bold table cell: "0.25 square metres for each face" carries a
#: section-shaped number and lowercase prose.
CLAUSE_SAINTJOHN = re.compile(r"^([1-9]\d?\.\d+(?:\(\d+[a-z]?\))?)\s+([A-Z\[].*)$")

#: The part heading: "5 General Provisions: Accessory Buildings and
#: Structures". Saint John's running header is a tab index rather than a
#: part name, so the part is only ever named here.
SUBSECTION_SAINTJOHN = re.compile(r"^([1-9]\d?)\s+([A-Z][A-Za-z].{3,})$")


#: "1.1. TITLE", "4.22. SIGNS" - CBRM. The trailing period is part of the
#: heading, so a pattern anchored on "N.N " never matched one and all 75
#: of the bylaw's headings were invisible: the document parsed as a
#: single 89,904-character clause.
CLAUSE_CBRM = re.compile(r"^(\d{1,2}\.\d{1,2})\.\s+(\S.*)$")

#: "2. DEFINITIONS" - the part heading, one number and a caps title.
SUBSECTION_CBRM = re.compile(r"^(\d{1,2})\.\d?\s+([A-Z][A-Z0-9 &/,'\u2019()\-\.]{3,})$")


#: St. John's changes structure halfway through. The general provisions
#: are numbered "1.1" and "3.2.1"; the zone chapters that follow are
#: headed by the zone itself and their provisions are "(1) PERMITTED
#: USES". Recognising only the first form left the whole second half -
#: 98,111 characters of zone standards - inside one clause called
#: CONFLICTING PROVISIONS.
CLAUSE_STJOHNS = re.compile(
    r"^(?:SECTION\s+(?P<part>\d{1,2})\s*[\u2013-]\s*(?P<parttitle>\S.*)"
    r"|(?P<number>\d{1,2}(?:\.\d{1,2}){1,2}|\(\d{1,2}\))\s+(?P<title>\S.*))$"
)

#: "MINI HOME PARK (MHP) ZONE". The code in the brackets is what the rest
#: of the bylaw cites, so it becomes the number and the whole heading the
#: title.
SUBSECTION_STJOHNS = re.compile(
    r"^(?P<title>[A-Z][A-Z0-9 ,\-/&]*)\((?P<number>[A-Z0-9\-]{1,6})\)\s*ZONE$"
)


#: "15.1", and "1." for the province-wide standards regulations appended
#: to the bylaw, which restart at 1 with their own numbering. Summerside
#: sets the heading above the number rather than beside it, so the number
#: line is not bold and the scheme reads its title from the line above.
CLAUSE_SUMMERSIDE = re.compile(r"^(\d{1,2}\.\d{1,2}|\d{1,2}\.)\s+(\S.*)$")

#: "Low Density Residential (R1) Zone" - the heading of a zone chapter.
#: Made the parent of everything under it, not the title of its first
#: clause: section 15.2 says "Up to four dwelling units are permitted on
#: each lot" and never names R1, so without this the chunk carries no
#: indication of which zone it governs - and a question about the R1 zone
#: cannot reach it.
SUBSECTION_SUMMERSIDE = re.compile(
    r"^(?P<title>.{3,60}?\((?P<number>[A-Za-z0-9]{1,5})\)\s*Zone)$",
    re.IGNORECASE,
)


SCHEMES: dict[str, NumberingScheme] = {
    "paren": NumberingScheme(),
    "moncton": NumberingScheme(
        clause=CLAUSE_MONCTON,
        subsection=SUBSECTION_DIVISION,
        clause_titles=False,
    ),
    "saintjohn": NumberingScheme(
        clause=CLAUSE_SAINTJOHN,
        subsection=SUBSECTION_SAINTJOHN,
    ),
    "cbrm": NumberingScheme(
        clause=CLAUSE_CBRM,
        subsection=SUBSECTION_CBRM,
        # Rows are named after the use rather than numbered.
        matrix_named_rows=True,
        matrix_min_columns=3,
        # CBRM's zone codes carry a trailing figure - UR1, RR5, R6 - which
        # the default pattern rejects, so the zone header of every use
        # table went unrecognised and a row of "P"s was taken for the
        # column headings instead.
        zone_code=re.compile(r"^[A-Z]{1,4}\d{0,2}(?:-\d+)?$"),
    ),
    "stjohns": NumberingScheme(
        clause=CLAUSE_STJOHNS,
        subsection=SUBSECTION_STJOHNS,
        # "(1) PERMITTED USES" repeats in all 43 zone chapters, so the
        # zone code is carried into the citation to keep it unique and
        # findable: MHP(1) rather than (1).
        clause_number_takes_parent=True,
    ),
    "summerside": NumberingScheme(
        clause=CLAUSE_SUMMERSIDE,
        subsection=SUBSECTION_SUMMERSIDE,
        # The heading sits on the line above the number, as in Moncton.
        clause_titles=False,
    ),
    "dotted3": NumberingScheme(
        clause=CLAUSE_DOTTED3,
        subsection=SUBSECTION_DOTTED,
    ),
    "dotted2": NumberingScheme(
        clause=CLAUSE_DOTTED2,
        subsection=SUBSECTION_NUMBER,
    ),
}

DEFAULT_SCHEME = "paren"


def scheme_for(name: str | None) -> NumberingScheme:
    """Look up a named scheme, falling back to Fredericton's."""
    return SCHEMES.get(name or DEFAULT_SCHEME, SCHEMES[DEFAULT_SCHEME])


# ---------------------------------------------------------------------
#  Bilingual documents (Moncton)
#
#  Fredericton publishes separate English and French PDFs, so language is
#  a property of the file. Moncton publishes ONE document with English in
#  a left column and French in a right column, on the same lines. Parsing
#  it as a single stream interleaves the two languages inside every
#  clause: an English question retrieves a chunk that is half French, and
#  a French citation quotes English text.
#
#  Pages are therefore split at the gutter and each side parsed as its own
#  document. The detector is a word that CROSSES the gutter: two prose
#  columns never do, while a full-width table row usually does. That
#  distinction matters because Moncton's lot-requirement tables span the
#  page - splitting one would sever a zone column from its setback values,
#  the same corruption the Fredericton sign matrix suffered.
# ---------------------------------------------------------------------

#: Where to look for the gutter, as a fraction of page width.
GUTTER_BAND = (0.40, 0.65)

#: Minimum gap that marks the bilingual gutter. Deliberately smaller than
#: GUTTER_MIN_WIDTH: Moncton sets its two languages only ~18pt apart, and
#: at the 30pt used for use-list columns the detector skipped the real
#: gutter at x=320 and locked onto a sparser gap at x=343 - which put
#: English text into the French corpus.
BILINGUAL_GUTTER_MIN_GAP = 14.0

#: Above this fraction of words crossing the gutter, the page is not two
#: columns of prose and must not be split.
MAX_CROSSING_RATIO = 0.02

#: Each side needs at least this many words to count as a real column.
MIN_COLUMN_WORDS = 20


def detect_bilingual_gutter(pdf, sample: int = 40) -> float | None:
    """The x where the second language column starts, or None.

    Found from the most common start of a wide gap in the middle band of
    the page, sampled across the document rather than taken from one page
    that might be a table.
    """
    starts: collections.Counter[int] = collections.Counter()
    pages = pdf.pages
    step = max(1, len(pages) // sample)

    for page in pages[: len(pages) : step]:
        low, high = page.width * GUTTER_BAND[0], page.width * GUTTER_BAND[1]
        for line in _group_lines(page.extract_words(extra_attrs=["fontname"])):
            for left, right in zip(line.words, line.words[1:]):
                gap = right["x0"] - left["x1"]
                if gap >= BILINGUAL_GUTTER_MIN_GAP and low <= right["x0"] <= high:
                    starts[round(right["x0"])] += 1

    if not starts:
        return None
    return float(starts.most_common(1)[0][0]) - COLUMN_X_TOLERANCE


def split_at_gutter(
    words: list[dict], gutter: float
) -> tuple[list[dict], list[dict]] | None:
    """Split one page's words into left and right columns.

    Returns None when the page is not two columns of prose - a cover page,
    a contents listing, or a full-width table - so the caller can handle
    it rather than cutting a table in half.
    """
    if len(words) < MIN_COLUMN_WORDS * 2:
        return None

    crossing = sum(1 for w in words if w["x0"] < gutter < w["x1"])
    if crossing / len(words) > MAX_CROSSING_RATIO:
        return None

    left = [w for w in words if w["x1"] <= gutter]
    right = [w for w in words if w["x0"] >= gutter]
    if len(left) < MIN_COLUMN_WORDS or len(right) < MIN_COLUMN_WORDS:
        return None

    return left, right



#: A prose line needs this many words on each side of the gutter. A
#: two-cell table row ("Minimum lot area | 558 m2") has fewer, and must
#: not be torn into two languages.
MIN_LINE_WORDS_PER_SIDE = 3


#: A section number standing in its own indented column.
INDENT_NUMBER = re.compile(r"^\d{1,3}(?:\.\d+)?(?:\s?\(\d+\))?$")


def _wide_gaps(words: list[dict]) -> list[tuple[float, float]]:
    """Horizontal gaps in one line wide enough to be a column boundary.

    A gap that follows a bare section number is not one. Moncton sets the
    number in its own indented column, so the first line of every section
    carries such a gap in each language - "158 <gap> Les aménagements" -
    and counting it as a column boundary makes the line look like a table
    row and leaves the two languages interleaved.
    """
    ordered = sorted(words, key=lambda w: w["x0"])
    return [
        (left["x1"], right["x0"])
        for left, right in zip(ordered, ordered[1:])
        if right["x0"] - left["x1"] >= BILINGUAL_GUTTER_MIN_GAP
        and not INDENT_NUMBER.match(left["text"])
    ]


def _line_is_bilingual_prose(words: list[dict], gutter: float) -> bool:
    """Whether one line is English prose beside its French translation.

    Three conditions, and a table row fails at least one. It must have a
    wide gap spanning the gutter; it must have no wide gap to the RIGHT
    of the gutter, since a table row's cells go on producing gaps across
    the rest of the page; and both sides must hold enough words to be
    prose rather than a label and its value.

    Getting this wrong in the permissive direction severs a lot
    requirement from its zone - the Fredericton sign-matrix corruption
    again - so every uncertain line is left whole.
    """
    gaps = _wide_gaps(words)
    spanning = [g for g in gaps if g[0] <= gutter <= g[1]]
    if len(spanning) != 1:
        return False
    if any(start > gutter for start, _ in gaps):
        return False

    left = sum(1 for w in words if w["x1"] <= gutter)
    right = sum(1 for w in words if w["x0"] >= gutter)
    return left >= MIN_LINE_WORDS_PER_SIDE and right >= MIN_LINE_WORDS_PER_SIDE


def split_lines_at_gutter(
    words: list[dict], gutter: float
) -> tuple[list[dict], list[dict], int]:
    """Assign a mixed page's words to the two languages, line by line.

    Some pages carry a paragraph of two-column prose above a full-width
    table. The page as a whole cannot be split - the table crosses the
    gutter - but leaving it whole interleaves the languages inside the
    prose: "158 No development shall be permitted and no main 158 Les
    aménagements ne sont permis et les".

    So prose lines are split and table lines are given to both corpora,
    which is what a bilingual table already is: its cells read "10.5 m /
    10,5 m" in either language.
    """
    left: list[dict] = []
    right: list[dict] = []
    split = 0

    for line in _group_lines(words):
        if _line_is_bilingual_prose(line.words, gutter):
            left.extend(w for w in line.words if w["x1"] <= gutter)
            right.extend(w for w in line.words if w["x0"] >= gutter)
            split += 1
        else:
            left.extend(line.words)
            right.extend(line.words)

    return left, right, split

def parse_bilingual_pdf(
    path: str | Path,
    *,
    scheme: NumberingScheme | None = None,
    languages: tuple[str, str] = ("en", "fr"),
) -> dict[str, list[Clause]]:
    """Parse an interleaved bilingual bylaw into one corpus per language.

    Pages that cannot be split - tables whose cells carry both languages
    inline ("10.5 m / 10,5 m") - are given to BOTH corpora rather than
    dropped. They hold the lot requirements that setback and frontage
    questions depend on, and their content genuinely is the source text in
    either language.
    """
    scheme = scheme or NumberingScheme()
    left_lang, right_lang = languages

    per_language: dict[str, list[ParsedPage]] = {left_lang: [], right_lang: []}
    split_pages = shared_pages = mixed_pages = 0

    with pdfplumber.open(str(path)) as pdf:
        gutter = detect_bilingual_gutter(pdf)
        if gutter is None:
            raise ValueError(
                f"{path} has no detectable bilingual gutter; it is probably "
                "not an interleaved two-column document."
            )

        for raw in pdf.pages:
            words = raw.extract_words(extra_attrs=["fontname"])
            halves = split_at_gutter(words, gutter)

            if halves is None:
                left_words, right_words, split = split_lines_at_gutter(
                    words, gutter
                )

                if split == 0:
                    # Nothing here is two-column prose: a whole-page table,
                    # given to both corpora and cited on its own rather than
                    # merged into whichever clause happened to precede it.
                    shared = parse_page(raw, scheme)
                    shared.standalone = True
                    per_language[left_lang].append(shared)
                    per_language[right_lang].append(shared)
                    shared_pages += 1
                    continue

                # A prose paragraph sitting above a full-width table. Full
                # page width, because the table rows still need their
                # columns mapped across the whole page.
                per_language[left_lang].append(
                    parse_page(raw, scheme, words=left_words)
                )
                per_language[right_lang].append(
                    parse_page(raw, scheme, words=right_words)
                )
                mixed_pages += 1
                continue

            left_words, right_words = halves
            per_language[left_lang].append(
                parse_page(raw, scheme, words=left_words, page_width=gutter)
            )
            per_language[right_lang].append(
                parse_page(
                    raw, scheme, words=right_words, page_width=raw.width - gutter
                )
            )
            split_pages += 1

    corpora = {
        language: assemble_clauses(pages, scheme)
        for language, pages in per_language.items()
    }

    log.info(
        "bilingual_pdf_parsed",
        path=str(path),
        gutter=round(gutter, 1),
        split_pages=split_pages,
        shared_pages=shared_pages,
        mixed_pages=mixed_pages,
        clauses={lang: len(cs) for lang, cs in corpora.items()},
    )
    return corpora
