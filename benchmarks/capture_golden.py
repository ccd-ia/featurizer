"""Capture v0.5.2 aggregator semantics as golden values.

Runs every migratable subquery aggregator over the shared case matrix and
writes ``tests/fixtures/preagg_golden_values.json``. This is the frozen
reference the set-based rewrite is proven against — it MUST be captured before
any aggregator is touched, and never edited afterward without a documented
semantics-change decision (plan Phase 3f).
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

from . import _db, preagg_cases

try:
    featurizer_version = version("featurizer")
except PackageNotFoundError:  # pragma: no cover - always installed under uv
    featurizer_version = "unknown"

GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "preagg_golden_values.json"
)


def capture(conn: Any) -> Dict[str, Any]:
    """Run all cases and return the golden document."""
    cases = preagg_cases.cases()
    captured: Dict[str, Any] = {}
    # Seed per case: ``seed_fixture`` drops and replaces the tables, so a
    # per-(fixture, ts_type) cache would run later cases against whichever
    # fixture was seeded last. The fixtures are tiny; correctness over speed.
    for case in cases:
        _db.seed_fixture(conn, case["fixture"], case["ts_type"])
        cfg = preagg_cases.config(case["agg"], case["ts_type"], case["interval"])
        rows = _db.run_config(conn, cfg)
        captured[case["id"]] = _db.canonicalize(rows)
    return {
        "_meta": {
            "featurizer_version": featurizer_version,
            "subquery_aggregator_count": len(preagg_cases.subquery_aggregators()),
            "migratable_aggregator_count": len(preagg_cases.migratable_aggregators()),
            "case_count": len(cases),
            "note": (
                "Golden values frozen from the correlated-subquery path before "
                "the set-based rewrite (specs/correlated-subquery-aggregator-"
                "scaling.html). Do not edit without a documented semantics change."
            ),
        },
        "cases": captured,
    }


def main() -> None:
    conn = _db.connect()
    try:
        document = capture(conn)
    finally:
        conn.rollback()
        conn.close()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    meta = document["_meta"]
    print(
        f"wrote {GOLDEN_PATH.relative_to(GOLDEN_PATH.parent.parent.parent)}: "
        f"{meta['case_count']} cases, "
        f"{meta['migratable_aggregator_count']} migratable aggregators "
        f"({meta['subquery_aggregator_count']} subquery total), "
        f"featurizer {meta['featurizer_version']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
