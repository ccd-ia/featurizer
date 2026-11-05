"""Every migratable aggregator's values must equal the frozen v0.5.2 golden.

This is the correctness gate for the set-based aggregator rewrite (plan
``specs/correlated-subquery-aggregator-scaling.html``, Phase 3f). The golden
values in ``tests/fixtures/preagg_golden_values.json`` were captured from the
correlated-subquery path *before* any rewrite; each migrated family must
reproduce them exactly (integers) / within FP tolerance (floats), with a
byte-identical output column set.

Before any migration this trivially passes — it proves the capture/compare
harness round-trips. After each migration batch it is the stop-the-line check:
a mismatch means fix the rewrite, never the golden file.

The case spec is imported from :mod:`benchmarks.preagg_cases` (repo-root package,
outside the wheel) so capture and verification cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import _db, preagg_cases

pytestmark = pytest.mark.integration

_GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "preagg_golden_values.json"
)


def _load_golden() -> dict:
    if not _GOLDEN_PATH.exists():
        pytest.skip(
            "golden values not captured: run `just db-up && just bench-capture-golden`"
        )
    return json.loads(_GOLDEN_PATH.read_text())


_GOLDEN = _load_golden() if _GOLDEN_PATH.exists() else {"cases": {}, "_meta": {}}
_CASES = preagg_cases.cases()

# Cells a semantics ADR moved on purpose, applied over the frozen golden so the
# file itself stays as captured: {case id: {(pid, column): new value}}.
# ADR-0020 (#86): ngram_3_freq counts only complete trigrams, so the edge
# fixture's two-row groups (2, 3 and 5) have none.
_MOVED_BY_ADR: dict[str, dict[tuple[int, str], object]] = {
    f"ngram_3_freq|edge|{ts_type}|all": {
        (pid, "NGRAM_3_FREQ(c.cat)"): None for pid in (2, 3, 5)
    }
    for ts_type in ("date", "timestamp")
}


def _expected(case_id: str) -> list[dict] | None:
    rows = _GOLDEN["cases"].get(case_id)
    moved = _MOVED_BY_ADR.get(case_id, {})
    if rows is None or not moved:
        return rows
    out = [dict(row) for row in rows]
    for row in out:
        for (pid, column), value in moved.items():
            if row["pid"] == pid:
                assert column in row, f"{case_id}: no column {column}"
                row[column] = value
    return out


def test_golden_inventory_matches_live_registry() -> None:
    """The frozen golden must cover exactly today's migratable aggregator set.

    Catches inventory drift: a new subquery aggregator (or a reclassification)
    changes the live case list; if the golden wasn't re-captured this fails
    loudly instead of silently leaving the new primitive unverified.
    """
    meta = _GOLDEN.get("_meta", {})
    assert meta.get("migratable_aggregator_count") == len(
        preagg_cases.migratable_aggregators()
    ), "golden is stale vs the live registry — re-run capture-golden"
    assert meta.get("case_count") == len(_CASES)
    assert set(_GOLDEN["cases"]) == {c["id"] for c in _CASES}, (
        "golden case ids differ from the live case matrix — re-capture"
    )


def test_every_adr_move_changes_a_golden_cell() -> None:
    """An override that equals the golden it overrides is stale."""
    for case_id, moved in _MOVED_BY_ADR.items():
        golden = {row["pid"]: row for row in _GOLDEN["cases"][case_id]}
        for (pid, column), value in moved.items():
            assert golden[pid][column] != value, f"{case_id} pid {pid}: not a move"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_aggregator_values_match_golden(pg_conn, case) -> None:
    """Re-run one case and assert its result equals the frozen golden."""
    expected = _expected(case["id"])
    if expected is None:
        pytest.skip(f"no golden for {case['id']} — re-capture")
    _db.seed_fixture(pg_conn, case["fixture"], case["ts_type"])
    cfg = preagg_cases.config(case["agg"], case["ts_type"], case["interval"])
    actual = _db.canonicalize(_db.run_config(pg_conn, cfg))
    equal, reason = _db.values_equal(expected, actual)
    assert equal, f"{case['id']}: {reason}"
