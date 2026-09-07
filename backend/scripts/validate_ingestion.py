#!/usr/bin/env python
"""Golden-question regression check (Phase 5, Step 1).

Runs each golden question through the real retrieval path and asserts two
things:

  * the expected bylaw section came back, and
  * the text of that section survived parsing intact.

The second is the one that catches silent regressions. A parser change can
leave retrieval working perfectly while quietly corrupting what the clause
says - zone codes coming apart into single letters, a measurement detached
from its label, a superscript lost so "75 m²" reads "75 m 2". Retrieval
still finds the chunk; the chunk is just wrong. So the expected strings
here are the exact ones that known defects destroyed.

Nothing asserts on generated wording. The model rephrases run to run;
what must not vary is which bylaw text it was handed.

Usage
-----
    python scripts/validate_ingestion.py
    python scripts/validate_ingestion.py --municipality nb_fredericton
    python scripts/validate_ingestion.py --threshold 0.9
    python scripts/validate_ingestion.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.services.retrieval import HybridRetriever  # noqa: E402

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden_questions"

# How many fused results count as "retrieved". Deliberately larger than the
# prompt's context: a section ranked eighth is still found, and demanding
# rank 1 would fail on questions with several defensible answers.
DEFAULT_TOP_K = 8

DEFAULT_THRESHOLD = 0.85


@dataclass
class QuestionResult:
    id: str
    question: str
    passed: bool
    retrieved: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


@dataclass
class SuiteResult:
    municipality_id: str
    language: str
    results: list[QuestionResult] = field(default_factory=list)
    degraded: bool = False
    degraded_reason: str | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def score(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0


def load_suites(municipality: str | None) -> list[dict]:
    suites = []
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if municipality and data.get("municipality_id") != municipality:
            continue
        suites.append(data)
    return suites


async def check_question(
    retriever: HybridRetriever,
    municipality_id: str,
    language: str,
    spec: dict,
    top_k: int,
) -> tuple[QuestionResult, bool]:
    result = await retriever.retrieve(
        spec["question"], municipality_id, language=language, limit=top_k
    )
    chunks = result.chunks
    sections = [c.section_number for c in chunks]
    corpus = "\n".join(c.chunk_content for c in chunks)

    failures: list[str] = []

    # A question with no answer in the corpus must retrieve nothing, or the
    # model will be handed an unrelated clause and asked to make it fit.
    if spec.get("expect_fallback"):
        if chunks:
            failures.append(
                f"expected no retrieval, got {sections[:3]}"
            )
    else:
        expected = spec.get("expect_sections") or []
        mode = spec.get("match", "all")
        if mode == "any":
            if expected and not any(s in sections for s in expected):
                failures.append(f"none of {expected} retrieved; got {sections[:5]}")
        else:
            missing = [s for s in expected if s not in sections]
            if missing:
                failures.append(f"missing {missing}; got {sections[:5]}")

        for needle in spec.get("expect_text") or []:
            if needle not in corpus:
                failures.append(f"text not found in retrieved chunks: {needle!r}")

    return (
        QuestionResult(
            id=spec["id"],
            question=spec["question"],
            passed=not failures,
            retrieved=sections[:5],
            failures=failures,
        ),
        result.degraded,
    )


async def run_suite(
    retriever: HybridRetriever,
    suite: dict,
    top_k: int,
) -> SuiteResult:
    outcome = SuiteResult(
        municipality_id=suite["municipality_id"], language=suite.get("language", "en")
    )

    for spec in suite["questions"]:
        result, degraded = await check_question(
            retriever, outcome.municipality_id, outcome.language, spec, top_k
        )
        outcome.results.append(result)
        if degraded:
            outcome.degraded = True

    return outcome


def render(outcome: SuiteResult, threshold: float, verbose: bool) -> None:
    width = 78
    print()
    print("=" * width)
    print(f"GOLDEN QUESTIONS — {outcome.municipality_id} [{outcome.language}]")
    print("=" * width)

    for result in outcome.results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"{mark}  {result.id}")
        if verbose or not result.passed:
            print(f"      Q: {result.question}")
            print(f"      retrieved: {result.retrieved}")
        for failure in result.failures:
            print(f"      -> {failure}")

    print("-" * width)
    print(
        f"{outcome.passed}/{len(outcome.results)} passed "
        f"({outcome.score:.0%}), threshold {threshold:.0%}"
    )

    if outcome.degraded:
        # Reported loudly: a suite that passes without vector search has
        # not exercised semantic retrieval at all, so calling it a clean
        # pass would overstate what was checked.
        print()
        print(
            "WARNING: embeddings were unavailable, so these results came "
            "from keyword search alone. Semantic retrieval was NOT "
            "exercised - do not read this as a full pass."
        )


async def main_async(args: argparse.Namespace) -> int:
    suites = load_suites(args.municipality)
    if not suites:
        target = args.municipality or "any municipality"
        print(f"No golden-question file found for {target} in {GOLDEN_DIR}")
        return 2

    retriever = await HybridRetriever.create(get_settings())
    exit_code = 0

    for suite in suites:
        outcome = await run_suite(retriever, suite, args.top_k)
        render(outcome, args.threshold, args.verbose)

        if outcome.score < args.threshold:
            exit_code = 1
        elif outcome.degraded and args.strict:
            # --strict is for the pre-deploy gate, where "we could not
            # actually test semantic retrieval" is not good enough.
            exit_code = 1

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden-question eval set.")
    parser.add_argument("--municipality", help="restrict to one municipality id")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"fraction that must pass (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"fused results counted as retrieved (default {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if embeddings were unavailable, even when the score passes",
    )
    parser.add_argument("--verbose", action="store_true", help="show every question")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
