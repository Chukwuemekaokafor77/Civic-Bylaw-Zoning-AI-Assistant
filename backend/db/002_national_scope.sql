-- =====================================================================
--  Migration 002 - National scope
--
--  Scope change: the target is now all of Canada (10 provinces and 3
--  territories) rather than Atlantic Canada alone. Launch still happens
--  Atlantic-first, expanding jurisdiction by jurisdiction, so the
--  ingestion pipeline is proven on nine municipalities before it is
--  pointed at hundreds.
--
--  The schema needed no structural change for this: `provinces` was
--  always a reference table and `province_code` a plain 2-char foreign
--  key, so widening coverage is data, not a rewrite. Two adjustments
--  were required and are made below.
--
--  Idempotent and safe to re-run. Apply after 001_init_schema.sql.
-- =====================================================================

SET search_path = public, extensions;


-- ---------------------------------------------------------------------
--  1. The table is named `provinces` but must now also hold territories.
--
--  Renaming the table would break the FK on bylaw_chunks and
--  municipalities, the RPC return signatures, and the frontend's reads —
--  for a cosmetic gain. Instead the table keeps its name and gains a
--  `type` column, so a UI can label Yukon correctly without pretending
--  it is a province.
-- ---------------------------------------------------------------------

ALTER TABLE provinces
    ADD COLUMN IF NOT EXISTS type VARCHAR(10) NOT NULL DEFAULT 'province';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'provinces_type_chk'
    ) THEN
        ALTER TABLE provinces
            ADD CONSTRAINT provinces_type_chk
            CHECK (type IN ('province', 'territory'));
    END IF;
END $$;


-- ---------------------------------------------------------------------
--  2. Seed the remaining jurisdictions.
--
--  The four Atlantic entries already exist from 001 and are updated
--  in place rather than duplicated. Names use the federal standard
--  short forms.
-- ---------------------------------------------------------------------

INSERT INTO provinces (code, name, type) VALUES
    ('AB', 'Alberta',                   'province'),
    ('BC', 'British Columbia',          'province'),
    ('MB', 'Manitoba',                  'province'),
    ('NB', 'New Brunswick',             'province'),
    ('NL', 'Newfoundland and Labrador', 'province'),
    ('NS', 'Nova Scotia',               'province'),
    ('ON', 'Ontario',                   'province'),
    ('PE', 'Prince Edward Island',      'province'),
    ('QC', 'Quebec',                    'province'),
    ('SK', 'Saskatchewan',              'province'),
    ('NT', 'Northwest Territories',     'territory'),
    ('NU', 'Nunavut',                   'territory'),
    ('YT', 'Yukon',                     'territory')
ON CONFLICT (code) DO UPDATE
    SET name = EXCLUDED.name,
        type = EXCLUDED.type;


-- ---------------------------------------------------------------------
--  3. Launch-readiness flag.
--
--  With 13 jurisdictions in the table, the frontend needs to know which
--  ones actually have indexed bylaws, or the province dropdown will
--  offer Nunavut and return nothing. This defaults to false everywhere
--  and is flipped per jurisdiction as its municipalities are ingested
--  and pass the Phase 5 golden-question eval.
--
--  Atlantic Canada is seeded true because its municipalities are the
--  launch set; a jurisdiction being live still depends on at least one
--  municipality having is_active = true.
-- ---------------------------------------------------------------------

ALTER TABLE provinces
    ADD COLUMN IF NOT EXISTS is_live BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE provinces SET is_live = TRUE  WHERE code IN ('NB', 'NS', 'PE', 'NL');
UPDATE provinces SET is_live = FALSE WHERE code NOT IN ('NB', 'NS', 'PE', 'NL');


-- ---------------------------------------------------------------------
--  4. Public read policy must respect is_live.
--
--  The 001 policy exposed every province row to anon. That was correct
--  when all four rows were launch jurisdictions; it is wrong now that
--  nine unlaunched ones exist.
-- ---------------------------------------------------------------------

DROP POLICY IF EXISTS provinces_public_read ON provinces;
CREATE POLICY provinces_public_read
    ON provinces FOR SELECT
    TO anon, authenticated
    USING (is_live);


-- ---------------------------------------------------------------------
--  NOT ADDRESSED HERE - flagged for a decision before those
--  jurisdictions are ingested:
--
--  * Language. bylaw_chunks.language and municipality_sources.language
--    are CHECK-constrained to ('en','fr'). That covers Quebec, whose
--    municipal by-laws are French-primary, and bilingual New Brunswick.
--    It does NOT cover the territories: Nunavut publishes in Inuktitut
--    and Inuinnaqtun, and the Northwest Territories recognises eleven
--    official languages. Ingesting NU or NT means widening that CHECK
--    and, more substantially, finding an embedding model that handles
--    those languages -- text-embedding-3-small will not do so usefully.
--
--  * Quebec terminology. Quebec uses "règlement de zonage" under a
--    different provincial planning statute, and its municipal structure
--    (MRCs, agglomerations, boroughs) does not map cleanly onto a flat
--    municipality list -- the same tenancy problem already flagged for
--    Halifax, at larger scale.
--
--  * Ontario. Many large municipalities publish zoning as searchable
--    HTML rather than PDF, which the Section 4 pipeline anticipates via
--    BeautifulSoup4 but which has not yet been exercised.
-- ---------------------------------------------------------------------
