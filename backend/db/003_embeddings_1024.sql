-- =====================================================================
--  Migration 003 - 1024-dimension embeddings
--
--  Why: the previous embedding provider's free tier allowed 1,000
--  embeddings per DAY. One municipality's bilingual bylaw alone is ~1,340
--  chunks, so the nine-municipality pilot was a week of trickling and the
--  national corpus was unreachable. Embeddings moved to a provider with a
--  200M-token grant and no daily cap.
--
--  A local model (BAAI/bge-m3) was trialled in between and rejected: it
--  cost 0.6s of CPU per query on every user request and separated a right
--  answer from a wrong one by only 0.049 cosine, against 0.081 hosted.
--  Both it and the chosen provider emit 1024 dimensions, so this
--  migration stands either way.
--
--  1024 instead of 1536 means the column, the HNSW index and both
--  retrieval RPCs have to change together. Nothing else in the schema is
--  affected: the natural key, RLS, the audit log and the keyword RPC's
--  text signals are all dimension-independent.
--
--  DESTRUCTIVE: every existing embedding is dropped. Vectors from two
--  different models are not comparable - a cosine distance between them
--  is a meaningless number that would still rank and still return
--  confident citations. Re-ingestion regenerates them; the chunk TEXT is
--  preserved, so only the vectors are rebuilt.
--
--  Apply after 002_national_scope.sql, then re-run ingestion with --force.
-- =====================================================================

SET search_path = public, extensions;

BEGIN;

-- ---------------------------------------------------------------------
--  1. Drop the RPCs first.
--
--  Both declare VECTOR(1536) in their signature or body, so they must go
--  before the column type changes rather than being left to fail at the
--  next call.
-- ---------------------------------------------------------------------

DROP FUNCTION IF EXISTS match_bylaw_chunks(VECTOR(1536), VARCHAR(64), VARCHAR(5), FLOAT, INT);
DROP FUNCTION IF EXISTS keyword_search_bylaw_chunks(TEXT, VARCHAR(64), VARCHAR(5), TEXT, INT);


-- ---------------------------------------------------------------------
--  2. Drop the vector index, then re-type the column.
--
--  The HNSW index is built for a fixed dimensionality; ALTER TYPE cannot
--  rewrite it in place.
-- ---------------------------------------------------------------------

DROP INDEX IF EXISTS idx_bylaw_chunks_embedding;

-- Cleared rather than cast. A 1536-dim vector cannot be reinterpreted as
-- 1024 dims, and silently truncating one would produce a vector that is
-- numerically valid and semantically meaningless.
UPDATE bylaw_chunks SET embedding = NULL WHERE embedding IS NOT NULL;

ALTER TABLE bylaw_chunks
    ALTER COLUMN embedding TYPE VECTOR(1024);

CREATE INDEX IF NOT EXISTS idx_bylaw_chunks_embedding
    ON bylaw_chunks USING hnsw (embedding vector_cosine_ops);


-- ---------------------------------------------------------------------
--  3. Clear the ingestion hashes.
--
--  A recorded hash means "this document is fully ingested". The chunk
--  rows survive but their vectors do not, so leaving the hashes in place
--  would make the next run report every document as unchanged and skip
--  the re-embedding this migration exists to force.
-- ---------------------------------------------------------------------

UPDATE municipality_sources SET content_hash = NULL;


-- ---------------------------------------------------------------------
--  4. Recreate the RPCs at 1024 dimensions.
--
--  Bodies are unchanged from 001 apart from the vector width.
-- ---------------------------------------------------------------------

CREATE FUNCTION match_bylaw_chunks (
  query_embedding VECTOR(1024),
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
    SELECT s.id AS chunk_id, 1.00::float AS score
    FROM scoped s
    WHERE v_hint IS NOT NULL
      AND s.section_number = v_hint

    UNION ALL

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

    SELECT s.id AS chunk_id,
           (0.80 * (ts_rank_cd(s.content_tsv, v_query)
                    / (ts_rank_cd(s.content_tsv, v_query) + 1.0)))::float AS score
    FROM scoped s
    WHERE v_query IS NOT NULL
      AND numnode(v_query) > 0
      AND s.content_tsv @@ v_query

    UNION ALL

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

REVOKE ALL ON FUNCTION match_bylaw_chunks FROM anon, authenticated;
REVOKE ALL ON FUNCTION keyword_search_bylaw_chunks FROM anon, authenticated;

COMMIT;
