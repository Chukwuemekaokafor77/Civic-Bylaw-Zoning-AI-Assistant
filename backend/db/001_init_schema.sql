-- =====================================================================
--  Atlantic Canada Civic Bylaw & Zoning AI Assistant
--  Phase 1, Step 1 - Initial schema (Supabase / PostgreSQL 15+)
--
--  Scope locked in Phase 0:
--    * Bilingual (EN + FR) from day one
--    * All nine pilot municipalities at launch
--    * Local-only hosting for Phase 1
--
--  This script is idempotent and safe to re-run.
--  Sections marked [SPEC]  are Section 3 of the architecture spec.
--  Sections marked [ADDED] are additions/deviations - each carries a
--  rationale and can be removed independently if rejected.
-- =====================================================================


-- =====================================================================
--  PART 0 - EXTENSIONS                                          [SPEC]
-- =====================================================================

-- Supabase keeps extensions in a dedicated `extensions` schema rather than
-- `public`, and includes it in the default search_path. Creating the schema
-- first keeps this script portable to a plain PostgreSQL instance (local
-- dev, CI), where it would not otherwise exist.
CREATE SCHEMA IF NOT EXISTS extensions;

CREATE EXTENSION IF NOT EXISTS vector  WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA extensions;   -- fuzzy / keyword search

-- Required for the rest of this script, not just for tidiness. Supabase's
-- default search_path already includes `extensions`, but a plain
-- PostgreSQL session uses "$user", public — where `VECTOR(1536)`,
-- `gin_trgm_ops` and `vector_cosine_ops` all fail to resolve. Setting it
-- here keeps one script working on both.
SET search_path = public, extensions;

-- French stemming for bilingual full-text search ships with core
-- PostgreSQL as the 'french' text search configuration; no extension
-- is required.


-- =====================================================================
--  PART 1 - REFERENCE TABLES                                    [SPEC]
-- =====================================================================

CREATE TABLE IF NOT EXISTS provinces (
    code VARCHAR(2) PRIMARY KEY,          -- 'NB', 'NS', 'PE', 'NL'
    name VARCHAR(50) NOT NULL
);

INSERT INTO provinces (code, name) VALUES
('NB', 'New Brunswick'),
('NS', 'Nova Scotia'),
('PE', 'Prince Edward Island'),
('NL', 'Newfoundland and Labrador')
ON CONFLICT (code) DO NOTHING;


CREATE TABLE IF NOT EXISTS municipalities (
    id VARCHAR(64) PRIMARY KEY,           -- e.g. 'nb_fredericton', 'ns_halifax'
    province_code VARCHAR(2) REFERENCES provinces(code) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    source_bylaw_name VARCHAR(255) NOT NULL,
    source_url TEXT NOT NULL,
    languages VARCHAR(5)[] DEFAULT ARRAY['en'],
    is_active BOOLEAN DEFAULT TRUE,
    bylaw_last_verified_at DATE,          -- drives the dated disclaimer (Section 5, Rule 6)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- =====================================================================
--  PART 2 - PER-DOCUMENT SOURCE REGISTRY                       [ADDED]
--
--  Why: `municipalities` holds exactly one source_bylaw_name and one
--  source_url. Under the bilingual lock, NB municipalities publish an
--  English AND a French bylaw document at different URLs - one row
--  cannot represent both. This child table also gives Phase 2 Step 4
--  (change detection) somewhere to store the per-document content hash,
--  for which the spec schema has no column.
--
--  `municipalities.source_url` / `source_bylaw_name` are retained
--  verbatim per spec as the primary (English) display source.
--  `municipalities.bylaw_last_verified_at` remains the value shown in
--  the UI disclaimer; maintain it as the OLDEST last_verified_at across
--  that municipality's sources, so the disclaimer never over-claims
--  freshness.
-- =====================================================================

CREATE TABLE IF NOT EXISTS municipality_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    municipality_id VARCHAR(64) NOT NULL
        REFERENCES municipalities(id) ON DELETE CASCADE,
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    bylaw_name VARCHAR(255) NOT NULL,
    source_url TEXT NOT NULL,
    source_type VARCHAR(10) NOT NULL DEFAULT 'pdf',   -- 'pdf' | 'html'
    document_version VARCHAR(50),                     -- e.g. '2024-06'
    content_hash VARCHAR(64),                         -- sha256 of fetched bytes
    last_fetched_at TIMESTAMP WITH TIME ZONE,
    last_verified_at DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT municipality_sources_type_chk
        CHECK (source_type IN ('pdf', 'html')),
    CONSTRAINT municipality_sources_lang_chk
        CHECK (language IN ('en', 'fr'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_municipality_sources_natural
    ON municipality_sources (municipality_id, language, bylaw_name);


-- =====================================================================
--  PART 3 - BYLAW CHUNKS & EMBEDDINGS                    [SPEC + ADDED]
-- =====================================================================

CREATE TABLE IF NOT EXISTS bylaw_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    municipality_id VARCHAR(64) REFERENCES municipalities(id) ON DELETE CASCADE,
    province_code VARCHAR(2) REFERENCES provinces(code) ON DELETE CASCADE,
    bylaw_name VARCHAR(255) NOT NULL,
    section_number VARCHAR(50) NOT NULL,
    section_title VARCHAR(255),
    chunk_content TEXT NOT NULL,
    language VARCHAR(5) DEFAULT 'en',
    page_number INT,
    source_document_version VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- [ADDED] A clause too long for one embedding gets split; chunk_index
    -- preserves order and completes the natural key below, so Phase 2
    -- re-ingestion can UPSERT instead of duplicating rows.
    chunk_index INT NOT NULL DEFAULT 0,

    CONSTRAINT bylaw_chunks_lang_chk CHECK (language IN ('en', 'fr'))
);

-- Tolerate re-runs against a table created before the [ADDED] columns.
ALTER TABLE bylaw_chunks
    ADD COLUMN IF NOT EXISTS chunk_index INT NOT NULL DEFAULT 0;

-- [ADDED] Language-aware full-text vector. See PART 6 rationale: trigram
-- similarity alone cannot carry keyword retrieval over long chunks.
-- Section title is weighted above body text. Added via ALTER so the
-- script stays re-runnable.
ALTER TABLE bylaw_chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(
            to_tsvector(
                CASE WHEN language = 'fr' THEN 'french'::regconfig
                     ELSE 'english'::regconfig END,
                coalesce(section_title, '')
            ), 'A')
        ||
        setweight(
            to_tsvector(
                CASE WHEN language = 'fr' THEN 'french'::regconfig
                     ELSE 'english'::regconfig END,
                chunk_content
            ), 'B')
    ) STORED;


-- =====================================================================
--  PART 4 - INDEXES
-- =====================================================================

-- [SPEC]
CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_trgm
    ON bylaw_chunks USING gin (chunk_content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_municipality
    ON bylaw_chunks (municipality_id);

-- [ADDED] Every retrieval path filters municipality_id AND language
-- (bilingual lock), so this composite is the index actually used.
CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_muni_lang
    ON bylaw_chunks (municipality_id, language);

-- [ADDED] Section-number lookup for citation-style queries ("6.3").
CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_section
    ON bylaw_chunks (municipality_id, section_number);

-- [ADDED] Full-text index backing the keyword RPC.
CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_tsv
    ON bylaw_chunks USING gin (content_tsv);

-- [ADDED] Vector index. The spec defines no index on `embedding`, which
-- leaves match_bylaw_chunks a sequential scan over every chunk in the
-- table. HNSW with cosine ops matches the `<=>` operator the RPC uses.
CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_embedding
    ON bylaw_chunks USING hnsw (embedding vector_cosine_ops);

-- [ADDED] Natural key enabling idempotent UPSERT during re-ingestion.
-- source_document_version is deliberately EXCLUDED: when a municipality
-- publishes a new version of a bylaw, the chunk must be REPLACED, not
-- duplicated alongside the stale one.
CREATE UNIQUE INDEX IF NOT EXISTS idx_bylaw_chunks_natural_key
    ON bylaw_chunks (municipality_id, bylaw_name, section_number, language, chunk_index);


-- =====================================================================
--  PART 5 - QUERY AUDIT LOG                              [SPEC + ADDED]
-- =====================================================================

CREATE TABLE IF NOT EXISTS query_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    municipality_id VARCHAR(64),
    user_query TEXT NOT NULL,
    retrieved_chunk_ids UUID[],
    response_text TEXT,
    was_fallback BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- [ADDED] Language the answer was produced in. Without it the
    -- Section 7 coverage dashboard cannot tell an FR gap from an EN gap.
    language VARCHAR(5)
);

ALTER TABLE query_log ADD COLUMN IF NOT EXISTS language VARCHAR(5);

CREATE INDEX IF NOT EXISTS idx_query_log_muni_created
    ON query_log (municipality_id, created_at DESC);

-- Fast "which topics hit the not-found fallback" scan (Phase 6 Step 4).
CREATE INDEX IF NOT EXISTS idx_query_log_fallback
    ON query_log (created_at DESC) WHERE was_fallback;


-- =====================================================================
--  PART 6 - RETRIEVAL RPCs
--
--  Both functions gain a `target_language` parameter. This is required
--  by the Phase 0 bilingual lock: without it a French chunk can be
--  retrieved for an English question and cited in an English answer,
--  which breaks Section 5 Rule 3 (citation integrity) and Rule 2 (no
--  mixing of sources). NULL means "no language filter".
--
--  Both are SECURITY INVOKER (the plpgsql default, stated explicitly).
--  The FastAPI backend calls them with the service-role key, which
--  bypasses RLS; anon/authenticated are revoked at the bottom of this
--  file, so a browser cannot call them directly.
-- =====================================================================

-- Drop the spec's original signatures if a previous run created them.
DROP FUNCTION IF EXISTS match_bylaw_chunks(VECTOR(1536), VARCHAR(64), FLOAT, INT);
DROP FUNCTION IF EXISTS match_bylaw_chunks(VECTOR(1536), VARCHAR(64), VARCHAR(5), FLOAT, INT);
DROP FUNCTION IF EXISTS keyword_search_bylaw_chunks(TEXT, VARCHAR(64), INT);
DROP FUNCTION IF EXISTS keyword_search_bylaw_chunks(TEXT, VARCHAR(64), VARCHAR(5), TEXT, INT);


-- ---------------------------------------------------------------------
--  6a. Vector search                                     [SPEC + ADDED]
--      Spec body preserved; adds target_language filter and returns
--      `language` so the caller can label a citation's source document.
-- ---------------------------------------------------------------------

CREATE FUNCTION match_bylaw_chunks (
  query_embedding VECTOR(1536),
  target_municipality VARCHAR(64),
  target_language VARCHAR(5) DEFAULT NULL,
  match_threshold FLOAT DEFAULT 0.3,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  municipality_id VARCHAR(64),
  province_code VARCHAR(2),
  bylaw_name VARCHAR(255),
  section_number VARCHAR(50),
  section_title VARCHAR(255),
  chunk_content TEXT,
  language VARCHAR(5),
  page_number INT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id,
    c.municipality_id,
    c.province_code,
    c.bylaw_name,
    c.section_number,
    c.section_title,
    c.chunk_content,
    c.language,
    c.page_number,
    c.metadata,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM bylaw_chunks c
  WHERE c.municipality_id = target_municipality
    AND (target_language IS NULL OR c.language = target_language)
    AND c.embedding IS NOT NULL
    AND 1 - (c.embedding <=> query_embedding) > match_threshold
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;


-- ---------------------------------------------------------------------
--  6b. Keyword / exact section-number search             [SPEC + ADDED]
--
--  Deviation from the spec body, and why:
--
--  The spec ranks by `similarity(chunk_content, search_text)` and gates
--  on `chunk_content % search_text`. Trigram similarity is a ratio over
--  the union of both strings' trigrams, so comparing a ~6-word question
--  against a ~200-word clause yields a score near zero - below the 0.3
--  default threshold that `%` enforces. The gate therefore almost never
--  fires, and `gin_trgm_ops` cannot accelerate an ORDER BY on
--  similarity() anyway. In practice nearly every hit would come from
--  `section_number = search_text`, which is strict equality: a user
--  asking about "6.3" would not match a chunk stored as "6.3(1)(a)".
--
--  This version keeps the trigram signal (lowest weight, still using
--  pg_trgm per spec) and adds two signals the spec's Section 2 already
--  sanctions ("Postgres full-text or trigram"):
--
--    1.00  exact section number match
--    0.90  section prefix match on a non-digit boundary
--          ("6.3" matches "6.3.2" and "6.3(1)(a)" but NOT "6.30")
--    ~0.80 full-text rank, language-aware (english/french stemming)
--    ~0.50 trigram similarity on chunk_content (spec's original signal)
--
--  Scores are bounded so the tiers never invert, which keeps the
--  Phase 3 reciprocal-rank-fusion merge predictable.
--
--  `section_hint` is optional and supplied by the backend after pulling
--  a citation-like token out of the question ("what does section 6.3
--  say" -> "6.3"). When NULL, only the text signals run.
--
--  Return shape is widened to match 6a so the Phase 3 merge can build a
--  full citation without a second round-trip per chunk.
-- ---------------------------------------------------------------------

CREATE FUNCTION keyword_search_bylaw_chunks (
  search_text TEXT,
  target_municipality VARCHAR(64),
  target_language VARCHAR(5) DEFAULT NULL,
  section_hint TEXT DEFAULT NULL,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  municipality_id VARCHAR(64),
  province_code VARCHAR(2),
  bylaw_name VARCHAR(255),
  section_number VARCHAR(50),
  section_title VARCHAR(255),
  chunk_content TEXT,
  language VARCHAR(5),
  page_number INT,
  metadata JSONB,
  rank FLOAT
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, extensions
AS $$
DECLARE
  v_config regconfig := CASE WHEN target_language = 'fr'
                             THEN 'french'::regconfig
                             ELSE 'english'::regconfig END;
  -- Guarded so an empty search_text never reaches websearch_to_tsquery,
  -- which raises a NOTICE on empty input. A section-number-only request
  -- (hint set, no prose) is a legitimate call and must stay quiet.
  v_text   TEXT      := nullif(btrim(coalesce(search_text, '')), '');
  v_query  tsquery   := CASE WHEN nullif(btrim(coalesce(search_text, '')), '') IS NULL
                             THEN NULL
                             ELSE websearch_to_tsquery(v_config, search_text) END;
  v_hint   TEXT      := nullif(btrim(coalesce(section_hint, '')), '');
BEGIN
  RETURN QUERY
  WITH scoped AS (
    SELECT c.*
    FROM bylaw_chunks c
    WHERE c.municipality_id = target_municipality
      AND (target_language IS NULL OR c.language = target_language)
  ),
  scored AS (
    -- Tier 1: exact section number
    SELECT s.id AS chunk_id, 1.00::float AS score
    FROM scoped s
    WHERE v_hint IS NOT NULL
      AND s.section_number = v_hint

    UNION ALL

    -- Tier 2: section prefix on a non-digit boundary
    SELECT s.id AS chunk_id, 0.90::float AS score
    FROM scoped s
    WHERE v_hint IS NOT NULL
      AND s.section_number <> v_hint
      AND left(s.section_number, length(v_hint)) = v_hint
      AND (
            length(s.section_number) = length(v_hint)
            OR substr(s.section_number, length(v_hint) + 1, 1) !~ '[0-9]'
          )

    UNION ALL

    -- Tier 3: language-aware full text, normalised into (0, 0.80)
    SELECT s.id AS chunk_id,
           (0.80 * (ts_rank_cd(s.content_tsv, v_query)
                    / (ts_rank_cd(s.content_tsv, v_query) + 1.0)))::float AS score
    FROM scoped s
    -- numnode() rather than a comparison against ''::tsquery: the empty
    -- tsquery literal raises a NOTICE on every parse, which would flood
    -- the logs once this RPC runs on every user question.
    WHERE v_query IS NOT NULL
      AND numnode(v_query) > 0
      AND s.content_tsv @@ v_query

    UNION ALL

    -- Tier 4: trigram similarity (the spec's original signal, retained
    -- as a low-weight fuzzy fallback for typos and OCR noise)
    SELECT s.id AS chunk_id,
           (0.50 * similarity(s.chunk_content, v_text))::float AS score
    FROM scoped s
    WHERE v_text IS NOT NULL
      AND s.chunk_content % v_text
  ),
  best AS (
    SELECT scored.chunk_id, max(scored.score) AS score
    FROM scored
    GROUP BY scored.chunk_id
  )
  SELECT
    c.id,
    c.municipality_id,
    c.province_code,
    c.bylaw_name,
    c.section_number,
    c.section_title,
    c.chunk_content,
    c.language,
    c.page_number,
    c.metadata,
    b.score AS rank
  FROM best b
  JOIN bylaw_chunks c ON c.id = b.chunk_id
  ORDER BY b.score DESC, c.section_number ASC
  LIMIT match_count;
END;
$$;


-- =====================================================================
--  PART 7 - ROW LEVEL SECURITY
--
--  Model: the FastAPI backend uses the service-role key, which bypasses
--  RLS entirely. These policies therefore govern ONE thing - what a
--  browser holding the anon key may read directly from Supabase.
--
--  Decision reflected here (flag if you want it otherwise):
--    * provinces + municipalities  -> readable by anon, so
--      RegionalSelector.tsx can populate the cascading dropdown without
--      a FastAPI round-trip. Only is_active municipalities are exposed,
--      which is what gates a municipality's launch in Phase 6 Step 2.
--    * municipality_sources        -> backend only (contains scrape
--      URLs, hashes and fetch state; no UI needs it).
--    * bylaw_chunks                -> backend only. All retrieval is
--      mediated by /stream so that every answer is audited and every
--      chunk reaches the user through a cited response.
--    * query_log                   -> backend only. Contains raw user
--      questions; must never be publicly readable or writable.
-- =====================================================================

ALTER TABLE provinces            ENABLE ROW LEVEL SECURITY;
ALTER TABLE municipalities       ENABLE ROW LEVEL SECURITY;
ALTER TABLE municipality_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE bylaw_chunks         ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_log            ENABLE ROW LEVEL SECURITY;

-- Public read: provinces
DROP POLICY IF EXISTS provinces_public_read ON provinces;
CREATE POLICY provinces_public_read
    ON provinces FOR SELECT
    TO anon, authenticated
    USING (true);

-- Public read: active municipalities only
DROP POLICY IF EXISTS municipalities_public_read ON municipalities;
CREATE POLICY municipalities_public_read
    ON municipalities FOR SELECT
    TO anon, authenticated
    USING (is_active);

-- No policies on municipality_sources, bylaw_chunks, query_log:
-- with RLS enabled and no permissive policy, anon/authenticated see
-- zero rows. service_role bypasses RLS and retains full access.

-- Belt and braces: remove table privileges as well, so a future
-- permissive policy added by mistake still cannot expose these.
REVOKE ALL ON municipality_sources FROM anon, authenticated;
REVOKE ALL ON bylaw_chunks         FROM anon, authenticated;
REVOKE ALL ON query_log            FROM anon, authenticated;

GRANT SELECT ON provinces      TO anon, authenticated;
GRANT SELECT ON municipalities TO anon, authenticated;

-- Explicit backend grants. Supabase already grants these to service_role
-- by default, but stating them keeps the script correct when applied to a
-- plain PostgreSQL instance (local dev, CI, or the Phase 5 test database)
-- where the REVOKEs above would otherwise leave the backend with nothing.
GRANT ALL ON provinces            TO service_role;
GRANT ALL ON municipalities       TO service_role;
GRANT ALL ON municipality_sources TO service_role;
GRANT ALL ON bylaw_chunks         TO service_role;
GRANT ALL ON query_log            TO service_role;

-- Retrieval RPCs are backend-only. Supabase grants EXECUTE to PUBLIC on
-- new functions in the public schema by default; revoke it explicitly.
REVOKE ALL ON FUNCTION match_bylaw_chunks(VECTOR(1536), VARCHAR(64), VARCHAR(5), FLOAT, INT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION keyword_search_bylaw_chunks(TEXT, VARCHAR(64), VARCHAR(5), TEXT, INT)
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION match_bylaw_chunks(VECTOR(1536), VARCHAR(64), VARCHAR(5), FLOAT, INT)
    TO service_role;
GRANT EXECUTE ON FUNCTION keyword_search_bylaw_chunks(TEXT, VARCHAR(64), VARCHAR(5), TEXT, INT)
    TO service_role;
