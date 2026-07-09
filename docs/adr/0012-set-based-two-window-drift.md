# 0012 — Set-based migration of the two-window drift aggregators

**Status:** Accepted
**Date:** 2026-07-09
**Deciders:** Adolfo De Unánue
**Extends:** [ADR-0010](0010-set-based-preaggregation.md) (reverses its "two-window drift is a non-goal" scope note)

## Context

ADR-0010 migrated 27 correlated-subquery aggregators to set-based companion CTEs
but explicitly deferred the two-window **drift** families (`kl_drift`,
`wasserstein_drift`) as a non-goal, on the assumption they "fire only under
special config."

Live-DB revalidation (`specs/feature-materialization-performance.html`,
`specs/reduce-child-stream-fanout-p3.html`) disproved that assumption. An
`EXPLAIN (ANALYZE)` of the dirtyduck all-agg matrix showed the entire 356.8s cost
was **9 correlated `SubPlan`s over `inspections_transform` at `loops=18909`** —
`kl_drift` (categorical) firing on dirtyduck's 3 categorical columns × 3 intervals.
It fires on ordinary categorical data, not just "special config", and being
O(target_rows × children) it dominated. Proof: dirtyduck all-agg **minus**
`{kl_drift, wasserstein_drift}` = 25.2s vs 356.8s. (An earlier "pure fan-out"
diagnosis was wrong; the fan-out — set-based companion CTEs — is cheap. Running
`EXPLAIN ANALYZE` earlier would have caught it.)

## Decision

**Migrate both drift families to the set-based pre-aggregation path** (the
ADR-0010 mechanism), closing the deferred scope:

- **`kl_drift`** — a single scan bounded to the two windows with
  `count(*) FILTER (WHERE recent)` / `FILTER (WHERE baseline)` derives the
  recent/baseline count per `(group, category)` in one pass; `rp`/`bp` are the
  per-window shares, and the reduction's `FILTER (rp>0 AND bp>0)` reproduces the
  correlated INNER-JOIN's shared support — **no self-join**.
- **`wasserstein_drift`** — one scan with each row tagged `is_recent`/`is_baseline`,
  and per-window quantiles taken with ordered-set `percentile_cont … FILTER`; an
  empty window yields a NULL percentile → NULL term, matching the correlated
  cross-join's NULL-on-empty behaviour.

Values are proven identical by the golden-value gate. The families were removed
from `benchmarks.preagg_cases.NEEDS_SPECIAL_CONFIG` and their **correlated** values
frozen as golden first (P3M cases added to the matrix — drift is 0 under P1M
because the baseline window is empty on the dense fixture, so only P3M exercises
the real two-window math). The set-based rewrite then reproduces them exactly.

## Consequences

- **Positive:** dirtyduck all-agg **356.8s → 27.6s (~13×)**, keeping all 272
  features — the ADR-0010 acceptance target (<120s on dirtyduck) now met on the
  *real* DB, not just synthetic 10k. Output column names/labels unchanged
  (ADR-0007). The migratable aggregator set is now 29; the golden is 232 cases.
- **Neutral:** the drift families now emit companion CTEs (kl_drift × categoricals
  + wasserstein × numerics × intervals), so the DB-free companion-CTE budget guard
  rose 132 → 144 — a correctness/perf win, not a fan-out regression.
- **Not a universal fix:** this resolved the *categorical two-window drift* cost
  (dirtyduck). donorschoose all-agg is unchanged (281s → 294s) — `kl_drift` does not
  fire there and its bottleneck is a *different*, un-diagnosed cost (1063 features,
  numeric-heavy). That needs its own `EXPLAIN (ANALYZE)` pass (run it first — the
  lesson of this whole investigation) and is tracked separately.
- **Remaining correlated non-goals:** the spatial families and the predicate-driven
  families (`bbox_area`, `distance_travelled`, `radius_of_gyration`, `spatial_std`,
  `first_passage_time`, `cross_type_latency`, `right_censoring_indicator`) genuinely
  need special config (coords / predicates) the standard fixture lacks and stay
  correlated; they did not appear in the live workloads. Migrate later only if a
  real workload demands it.
