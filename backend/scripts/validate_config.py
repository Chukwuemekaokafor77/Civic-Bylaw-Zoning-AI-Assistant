#!/usr/bin/env python3
"""Validate municipalities_config.json.

Runs in CI on every PR. The registry is the file outside contributors will
touch most often, and a bad entry there does not fail loudly at ingestion
time -- it silently indexes the wrong document, or the right document
under the wrong jurisdiction. This catches the structural mistakes.

Usage:
    python scripts/validate_config.py                # structure only
    python scripts/validate_config.py --check-urls   # also fetch every source_url
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONFIG = Path(__file__).parent / "municipalities_config.json"

# Mirrors provinces seeded by 002_national_scope.sql.
JURISDICTIONS = {
    "AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK",
    "NT", "NU", "YT",
}

# Mirrors the CHECK constraint on bylaw_chunks.language.
LANGUAGES = {"en", "fr"}

VALID_STATUSES = {
    "ready",
    "ready_high_staleness_risk",
    "blocked_needs_decision",
    "blocked_no_citable_source",
}


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    municipalities = data.get("municipalities", [])

    if not municipalities:
        errors.append("no municipalities defined")

    seen_ids: set[str] = set()

    for m in municipalities:
        mid = m.get("id", "<missing id>")

        if mid in seen_ids:
            errors.append(f"{mid}: duplicate id")
        seen_ids.add(mid)

        prov = m.get("province_code")
        if prov not in JURISDICTIONS:
            errors.append(f"{mid}: province_code {prov!r} is not a Canadian jurisdiction")

        # The id prefix must agree with province_code, otherwise a
        # municipality can be filed under the wrong jurisdiction and
        # still look plausible in the dropdown.
        if prov and not mid.startswith(f"{prov.lower()}_"):
            errors.append(
                f"{mid}: id should start with {prov.lower()}_ to match province_code {prov}"
            )

        langs = set(m.get("languages", []))
        if not langs:
            errors.append(f"{mid}: languages must not be empty")
        if unsupported := langs - LANGUAGES:
            errors.append(
                f"{mid}: language(s) {sorted(unsupported)} exceed the "
                f"bylaw_chunks.language CHECK constraint {sorted(LANGUAGES)}"
            )

        status = m.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{mid}: unknown status {status!r}")

        is_active = m.get("is_active")
        sources = m.get("sources", [])

        # An active municipality with no sources would be offered in the
        # UI and answer nothing.
        if is_active and not sources:
            errors.append(f"{mid}: is_active is true but it has no sources")

        # A blocked municipality must not be live.
        if status and status.startswith("blocked") and is_active:
            errors.append(f"{mid}: status is {status} but is_active is true")

        for s in sources:
            url = s.get("source_url", "")
            if not url.startswith("https://"):
                errors.append(f"{mid}: source_url is not https: {url!r}")
            if s.get("language") not in langs:
                errors.append(
                    f"{mid}: source language {s.get('language')!r} is not "
                    f"declared in the municipality's languages {sorted(langs)}"
                )
            if s.get("source_type") not in {"pdf", "html"}:
                errors.append(f"{mid}: source_type must be pdf or html")

        # Guard the disclaimer. A verification date that no human set is
        # worse than no date, because Section 5 Rule 6 shows it to the public.
        if m.get("bylaw_last_verified_at") and not m.get("verified_by"):
            errors.append(
                f"{mid}: bylaw_last_verified_at is set but verified_by is missing. "
                "This date is shown to the public as a freshness claim, so it must "
                "record who confirmed it."
            )

    return errors


def check_urls(data: dict) -> list[str]:
    """Fetch every source_url. Network-dependent, so it is opt-in."""
    import urllib.error
    import urllib.request

    problems: list[str] = []
    for m in data.get("municipalities", []):
        for s in m.get("sources", []):
            url = s["source_url"]
            req = urllib.request.Request(
                url,
                method="GET",
                headers={"User-Agent": "civic-bylaw-assistant/link-check", "Range": "bytes=0-0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    if resp.status not in (200, 206):
                        problems.append(f"{m['id']} [{s['language']}]: HTTP {resp.status} {url}")
                    else:
                        print(f"  ok   {m['id']} [{s['language']}] {resp.status}")
            except urllib.error.HTTPError as e:
                problems.append(f"{m['id']} [{s['language']}]: HTTP {e.code} {url}")
            except Exception as e:  # noqa: BLE001 - report any failure, don't crash the run
                problems.append(f"{m['id']} [{s['language']}]: {type(e).__name__} {url}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-urls", action="store_true", help="fetch every source_url")
    args = ap.parse_args()

    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    municipalities = data.get("municipalities", [])
    active = [m for m in municipalities if m.get("is_active")]
    print(f"{CONFIG.name}: {len(municipalities)} municipalities, {len(active)} active")

    errors = validate(data)

    if args.check_urls:
        print("\nchecking source URLs...")
        errors += check_urls(data)

    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("\nvalidation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
