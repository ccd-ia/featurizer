"""DB-free fan-out budget guard for the child-stream pre-aggregation tier (plan P3).

The full-aggregator config emits one companion CTE per ``(family, column,
interval)``, each an independent scan of the child stream — the fan-out that
dominates full-cohort materialization on real data. This pins the current
companion-CTE count for a representative synthetic config so:

- Phase 3's signature-merge is *proven* to reduce it (drop the ceiling as it lands), and
- a future change that re-inflates the fan-out trips loudly.

Uses the same synthetic config as ``benchmarks.fanout_report`` so the guard and
the report never drift. No database required.
"""

from __future__ import annotations

from benchmarks.fanout_report import fanout, synthetic_config

# Baseline: 3-categorical / 1-numeric child, 3 intervals, full default aggregator
# set. Raised 132 → 144 on 2026-07-09 when kl_drift / wasserstein_drift migrated
# from the correlated path to companion CTEs (kl_drift on 3 categoricals + ws on 1
# numeric × 3 intervals = 12 new companion CTEs — a correctness/perf win, not a
# fan-out regression). Never raise this without a deliberate reason.
COMPANION_CTE_BUDGET = 144


def test_synthetic_fanout_within_budget() -> None:
    m = fanout(synthetic_config())
    assert m["companion_ctes"] <= COMPANION_CTE_BUDGET, (
        f"companion-CTE fan-out {m['companion_ctes']} exceeds the budget "
        f"{COMPANION_CTE_BUDGET} — a change re-inflated the child-stream fan-out "
        "(plan P3 reduces it, never grows it)."
    )


def test_fanout_report_shape() -> None:
    """The report exposes the metrics the guard and the acceptance run depend on."""
    m = fanout(synthetic_config())
    assert set(m) == {"features", "total_ctes", "companion_ctes", "child_stream_refs"}
    assert m["features"] > 0 and m["companion_ctes"] > 0
    # Sanity: the fan-out shape holds — many more child-stream refs than features.
    assert m["child_stream_refs"] > m["features"]
