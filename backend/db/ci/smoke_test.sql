-- =====================================================================
--  CI retrieval smoke test
--
--  Runs against a throwaway database after the migrations are applied.
--  Every assertion here guards behaviour that can regress SILENTLY --
--  the query still returns rows, they are just the wrong rows, and a
--  wrong-but-well-formatted citation is the worst failure this system
--  can produce.
--
--  Fails loudly via RAISE EXCEPTION so the CI step exits non-zero.
-- =====================================================================

SET search_path = public, extensions;

-- Build a vector of the column's width without a real embedding call.
-- 1024 since migration 003 moved embeddings to a local model; this must
-- track bylaw_chunks.embedding or every insert here fails on dimension.
CREATE OR REPLACE FUNCTION pg_temp.mkvec(fill_val real, first_val real DEFAULT NULL)
RETURNS extensions.vector LANGUAGE sql IMMUTABLE AS $fn$
  SELECT replace(replace(
           (CASE WHEN first_val IS NULL
                 THEN array_fill(fill_val, ARRAY[1024])
                 ELSE array_prepend(first_val, array_fill(fill_val, ARRAY[1023]))
            END)::text, '{', '['), '}', ']')::extensions.vector;
$fn$;

INSERT INTO municipalities (id, province_code, name, source_bylaw_name, source_url, languages, is_active)
VALUES ('ci_muni_a', 'NB', 'CI Municipality A', 'Test By-law', 'https://example.invalid/a.pdf', ARRAY['en','fr'], TRUE),
       ('ci_muni_b', 'NS', 'CI Municipality B', 'Test By-law', 'https://example.invalid/b.pdf', ARRAY['en'],      TRUE);

INSERT INTO bylaw_chunks (municipality_id, province_code, bylaw_name, section_number, section_title, chunk_content, language, chunk_index, embedding)
VALUES
 ('ci_muni_a','NB','Test By-law','6.3','Secondary Suites',
  'A secondary suite is permitted in an R1 zone provided the minimum rear yard setback is 7.5 metres.','en',0,pg_temp.mkvec(0.1)),
 ('ci_muni_a','NB','Test By-law','6.3(1)(a)','Floor Area',
  'The gross floor area of a secondary suite shall not exceed 80 square metres.','en',0,pg_temp.mkvec(0.1,0.9)),
 ('ci_muni_a','NB','Test By-law','6.30','Signage',
  'No sign shall be erected within 3 metres of a street line.','en',0,pg_temp.mkvec(0.1,0.5)),
 ('ci_muni_a','NB','Arrete test','6.3','Logements secondaires',
  'Un logement secondaire est permis dans une zone R1 pourvu que la marge de recul arriere soit de 7,5 metres.','fr',0,pg_temp.mkvec(0.1,0.7)),
 ('ci_muni_b','NS','Other By-law','6.3','Other Municipality Section',
  'This clause belongs to a different municipality entirely.','en',0,pg_temp.mkvec(0.1,0.8));

DO $$
DECLARE
  n INT;
  r RECORD;
BEGIN
  -- 1. Exact section number outranks everything else.
  SELECT rank INTO n FROM keyword_search_bylaw_chunks('section 6.3','ci_muni_a','en','6.3',10)
   WHERE section_number = '6.3';
  IF n IS NULL THEN
    RAISE EXCEPTION 'exact section 6.3 was not returned';
  END IF;

  -- 2. A sub-clause of 6.3 must be reachable from the hint "6.3".
  IF NOT EXISTS (
    SELECT 1 FROM keyword_search_bylaw_chunks('section 6.3','ci_muni_a','en','6.3',10)
     WHERE section_number = '6.3(1)(a)'
  ) THEN
    RAISE EXCEPTION 'sub-clause 6.3(1)(a) not matched by hint 6.3';
  END IF;

  -- 3. But 6.30 is a DIFFERENT section and must not be dragged in.
  IF EXISTS (
    SELECT 1 FROM keyword_search_bylaw_chunks('section 6.3','ci_muni_a','en','6.3',10)
     WHERE section_number = '6.30'
  ) THEN
    RAISE EXCEPTION 'section 6.30 leaked into results for hint 6.3 (boundary check broken)';
  END IF;

  -- 4. Ranking order: exact must beat prefix.
  SELECT section_number INTO r FROM keyword_search_bylaw_chunks('section 6.3','ci_muni_a','en','6.3',10)
   ORDER BY rank DESC LIMIT 1;
  IF r.section_number <> '6.3' THEN
    RAISE EXCEPTION 'expected 6.3 ranked first, got %', r.section_number;
  END IF;

  -- 5. Language filter: an English request must never return French chunks.
  SELECT count(*) INTO n FROM keyword_search_bylaw_chunks('secondary suite','ci_muni_a','en',NULL,10)
   WHERE language <> 'en';
  IF n > 0 THEN
    RAISE EXCEPTION 'English keyword search returned % non-English chunk(s)', n;
  END IF;

  SELECT count(*) INTO n FROM match_bylaw_chunks(pg_temp.mkvec(0.1),'ci_muni_a','en',0.0,50)
   WHERE language <> 'en';
  IF n > 0 THEN
    RAISE EXCEPTION 'English vector search returned % non-English chunk(s)', n;
  END IF;

  -- 6. Multi-tenant isolation: never return another municipality's clauses.
  SELECT count(*) INTO n FROM keyword_search_bylaw_chunks('section 6.3','ci_muni_a','en','6.3',50)
   WHERE municipality_id <> 'ci_muni_a';
  IF n > 0 THEN
    RAISE EXCEPTION 'keyword search leaked % chunk(s) from another municipality', n;
  END IF;

  SELECT count(*) INTO n FROM match_bylaw_chunks(pg_temp.mkvec(0.1),'ci_muni_a',NULL,0.0,50)
   WHERE municipality_id <> 'ci_muni_a';
  IF n > 0 THEN
    RAISE EXCEPTION 'vector search leaked % chunk(s) from another municipality', n;
  END IF;

  -- 7. Degenerate input must not error (empty and stopword-only queries).
  PERFORM count(*) FROM keyword_search_bylaw_chunks('','ci_muni_a','en',NULL,5);
  PERFORM count(*) FROM keyword_search_bylaw_chunks(NULL,'ci_muni_a','en','6.3',5);
  PERFORM count(*) FROM keyword_search_bylaw_chunks('the and of','ci_muni_a','en',NULL,5);

  -- 8. Upsert on the natural key replaces rather than duplicates.
  INSERT INTO bylaw_chunks (municipality_id, province_code, bylaw_name, section_number, chunk_content, language, chunk_index, embedding)
  VALUES ('ci_muni_a','NB','Test By-law','6.3','AMENDED TEXT','en',0,pg_temp.mkvec(0.2))
  ON CONFLICT (municipality_id, bylaw_name, section_number, language, chunk_index)
  DO UPDATE SET chunk_content = EXCLUDED.chunk_content;

  SELECT count(*) INTO n FROM bylaw_chunks
   WHERE municipality_id='ci_muni_a' AND bylaw_name='Test By-law'
     AND section_number='6.3' AND language='en';
  IF n <> 1 THEN
    RAISE EXCEPTION 'upsert duplicated a chunk: found % rows for the same natural key', n;
  END IF;

  -- 9. RLS: anon must not read bylaw_chunks or query_log.
  BEGIN
    SET LOCAL ROLE anon;
    PERFORM count(*) FROM bylaw_chunks;
    RESET ROLE;
    RAISE EXCEPTION 'anon was able to read bylaw_chunks';
  EXCEPTION WHEN insufficient_privilege THEN
    RESET ROLE;  -- expected
  END;

  RAISE NOTICE 'all retrieval smoke tests passed';
END $$;

-- Leave the database clean in case later steps are added.
DELETE FROM bylaw_chunks   WHERE municipality_id LIKE 'ci_muni_%';
DELETE FROM municipalities WHERE id LIKE 'ci_muni_%';
