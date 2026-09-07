## What this changes

<!-- One or two sentences. -->

## Type

- [ ] Adds or updates a municipality in the registry
- [ ] Ingestion pipeline
- [ ] Retrieval / generation
- [ ] Frontend
- [ ] Schema migration
- [ ] Docs / tooling

## If this touches the municipality registry

- [ ] `python scripts/validate_config.py` passes
- [ ] `python scripts/validate_config.py --check-urls` passes
- [ ] Every source URL is hosted by the municipality itself
- [ ] I confirmed the document is the **current** consolidation, and said how in the description
- [ ] `bylaw_last_verified_at` is `null`, or names who verified it and when

## If this touches retrieval or the schema

- [ ] `backend/db/ci/smoke_test.sql` still passes
- [ ] Migrations are idempotent (applying twice is a no-op)
- [ ] Multi-tenant filtering is intact: no query can return another municipality's chunks

## Answer quality

Answers from this system are cited and look authoritative, so a wrong answer
carries further than a wrong answer normally would.

- [ ] This change cannot cause a clause to be attributed to the wrong municipality
- [ ] This change cannot cause a repealed or superseded bylaw to be served
- [ ] Citations still resolve to a real section in a real source document
