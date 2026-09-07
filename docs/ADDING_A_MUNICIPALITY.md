# Adding a municipality

This is the highest-leverage contribution to this project. Indexing a
municipality is mostly research, not code, and research parallelises across
people in a way that engineering does not.

It is also the contribution with the most room to do quiet damage. This
assistant answers questions people act on — whether they can add a secondary
suite, how far a deck must sit from a lot line. An answer citing a repealed
bylaw is worse than no answer, because the citation makes it look verified.

Read the verification section before anything else.

---

## The one rule

> **A URL that returns a PDF is not evidence that the PDF is in force.**

This is not hypothetical. While assembling the launch set we found:

- **Halifax Peninsula Land Use By-law** — still served from a live
  `halifax.ca` URL, HTTP 206, valid PDF. It was **repealed on 2021-10-26**
  and has not been in force since 2021-11-27. Anyone trusting HTTP status
  would have indexed a void bylaw.
- **Saint John** — the widely-cited 2021 zoning PDF is dead (404), and the
  2025 reform it predates merged the R1, R2 and RSS zones into a single RL
  zone. A stale copy answers secondary-suite questions using zones that no
  longer exist.
- **Charlottetown** — the City's own document links 404 or redirect to a
  login gateway, and the city is mid-rewrite of the bylaw. It remains
  unlaunched because no citable current source could be established.

Four of the URLs that search engines surfaced for the first nine
municipalities were unusable. Assume yours may be too.

---

## Step 1 — Find the source

Prefer, in order:

1. The municipality's own **office consolidation** of the zoning bylaw —
   the amended, currently-in-force text.
2. The municipality's zoning/planning page, following its own link.
3. Nothing else. Provincial regulator archives, third-party mirrors and
   `scribd`-style copies are not acceptable: they are snapshots of unknown
   currency hosted by someone who is not the legal authority.

Watch for municipalities that have **no single zoning bylaw**. Halifax has
roughly twenty-one Land Use By-laws, one per community plan area. Quebec
municipalities sit under a different provincial planning statute with
boroughs and agglomerations. If yours is like this, say so in the issue and
stop — it needs a modelling decision, not a config entry.

## Step 2 — Verify it is in force

Do at least one of:

- **Best:** email or call the municipality's Planning & Development
  department and ask "is this the current consolidation?" Record who
  answered and when.
- Compare the document's consolidation date against the most recent
  amendment listed on the municipality's own bylaws page. If amendments
  post-date the consolidation, note which ones are missing.
- Check for a repeal notice. A bylaw can be repealed while its PDF stays
  online indefinitely.

Then check the URL mechanically:

```bash
curl -sSL -o /dev/null -r 0-0 \
  -w '%{http_code} %{content_type} %{url_effective}\n' "<URL>"
```

Expect `200` or `206` and `application/pdf` or `text/html`. If
`url_effective` differs from your URL, the source redirects — record both:
cite the stable one, hash the resolved one.

## Step 3 — Add the entry

Edit `backend/scripts/municipalities_config.json`:

```json
{
  "id": "pe_summerside",
  "province_code": "PE",
  "name": "Summerside",
  "languages": ["en"],
  "is_active": true,
  "status": "ready",
  "source_bylaw_name": "Zoning Bylaw CS-40",
  "source_url": "https://...",
  "bylaw_last_verified_at": null,
  "sources": [
    {
      "language": "en",
      "bylaw_name": "Zoning Bylaw CS-40",
      "source_url": "https://...",
      "source_type": "pdf",
      "document_version": "Effective 2025-01-23",
      "http_status": 200,
      "content_type": "application/pdf",
      "http_checked_at": "2026-09-06"
    }
  ],
  "notes": "Anything the next person needs to know."
}
```

Conventions the validator enforces:

- `id` is `<province_code lowercased>_<municipality>`, e.g. `ns_cbrm`.
- `source_url` must be `https`.
- Every `sources[].language` must appear in the municipality's `languages`.
- `is_active` must be `false` for any `status` beginning `blocked`.
- An `is_active` municipality must have at least one source.

**Leave `bylaw_last_verified_at` as `null`** unless a human confirmed
currency with the municipality. That date is rendered directly into the
public disclaimer as a freshness claim. A guessed date there is a lie told
at scale.

### Bilingual municipalities

Two shapes exist, and they need different parsing:

- **Separate documents** (Fredericton): one PDF per language. Language is a
  property of the file. Add one `sources` entry per language.
- **Interleaved single document** (Moncton): English and French in one PDF.
  Both `sources` entries point at the same URL, and the parser must split
  and tag languages per chunk. Flag this loudly in `notes`.

`bylaw_chunks.language` is currently constrained to `en` and `fr`. That
covers Quebec and bilingual New Brunswick. It does **not** cover Nunavut
(Inuktitut, Inuinnaqtun) or the Northwest Territories (eleven official
languages) — those need both a schema change and an embedding model that
handles the languages, which `text-embedding-3-small` does not.

## Step 4 — Validate

```bash
cd backend
python scripts/validate_config.py              # structure
python scripts/validate_config.py --check-urls # fetch every source
```

Both must pass. CI runs the structural check on every PR.

## Step 5 — Ingest and evaluate

Once the ingestion pipeline exists (Phase 2):

```bash
python scripts/ingest_bylaws.py --municipality <id>
```

Then hand-check a sample of rows in `bylaw_chunks`. Look for:

- Section numbers parsed as section numbers, not swallowed into body text.
- Clauses that end where the clause ends, not mid-sentence.
- `page_number` matching the actual PDF page.
- Correct `language` on every chunk for bilingual sources.

Add 10–20 golden questions under `backend/tests/golden_questions/<id>.json`
pairing a realistic question with the section that should be cited. These
are the regression net; a municipality should not go live without them.

## Step 6 — Go live

`is_active` stays `false` until ingestion is verified and the golden
questions pass. A jurisdiction additionally needs `provinces.is_live = true`
before it appears in the UI at all.

---

## What gets a PR rejected

- A source URL that redirects to a login page.
- A `bylaw_last_verified_at` set without saying who verified it and how.
- A third-party or archive copy where an official one exists.
- Ingesting one plan area of a multi-plan-area municipality as if it covered
  the whole municipality. This produces confidently wrong answers for
  everyone outside that area, which is the exact failure this project exists
  to avoid.
