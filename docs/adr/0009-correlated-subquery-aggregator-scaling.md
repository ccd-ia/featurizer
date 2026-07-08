# 0009 — Correlated-subquery aggregators do not scale to full-cohort materialization

**Status:** Accepted (decision: spike-and-defer the remediation) — **follow-up RESOLVED 2026-07-08**
**Date:** 2026-07-06
**Deciders:** Adolfo De Unánue

> **Resolution (2026-07-08).** The deferred remediation is done. The spike
> ([ADR-0010](0010-set-based-preaggregation.md)) chose set-based pre-aggregation
> over the indexed-temp-table alternative and it was implemented across all 27
> migratable subquery aggregators (plan:
> `specs/correlated-subquery-aggregator-scaling.html`). The headline number is
> fixed: the synthetic 10k-parent all-aggregator matrix went from **>300 s
> (censored)** to **2.6 s**, with output names and values proven identical to
> v0.5.2 by a golden-value harness. The "bounded cohorts" guidance below applies
> now only to the special-config families still on the correlated path
> (predicate / drift / spatial); the migrated advanced tier scales like the
> default tier. See ADR-0010 for the decision and the before/after numbers.

## Context

26 of featurizer's aggregators are `SubqueryAggregator`s — the gap family,
`burstiness`, `entropy`, `hhi`, `gini`, the n-gram / sequence / Markov family,
`theil`, `median_absolute_deviation`, `trimmed_mean_10`, `variance_ratio`,
`acf_1`, `mean_deviation`, and others. Each emits a **correlated subquery per
target row** via the lateral join: for every target entity, the subquery
re-scans that entity's child stream. Cost is therefore
`O(target_rows × subqueries_selected × child_scan)`.

Measured against three live datasets (advanced-aggregator hardening session):

| variant | dataset | target rows | wall-clock |
|---|---|--:|--:|
| narrow (10 simple aggs) | dirtyduck | 22,169 | 3.6 s |
| narrow | chicago311 | 30,654 | 1.0 s |
| narrow | donorschoose | 3,000 | 1.6 s |
| all-agg (advanced set) | dirtyduck | 22,169 | **> 300 s (timeout)** |

The simple/default-active aggregators (plain `GROUP BY` reductions) materialize a
full cohort in seconds. The correlated-subquery tier does not: an all-agg
materialization over 20k–30k target rows exceeds a 300 s budget.

This is a *performance* property, not a correctness bug — the generated SQL is
valid and all aggregators execute (see
`tests/integration/test_all_aggregators_execution.py`). The correctness
hardening (division guards, degenerate-input NULLs, the planner empty-CTE fix,
`mean_deviation`/`skewness`/`kurtosis`/`harmonic_mean`) is done and shipped.

## Decision

**Spike-and-defer.** We do NOT rewrite the correlated-subquery aggregators in
this pass. The remediation — pre-aggregating each child stream once per group
into an intermediate the stat aggregators read, and/or LATERAL rewrites — is an
architectural change touching 26 aggregators, with real regression risk, and is
out of scope for a correctness-hardening effort. It gets its **own dedicated
implementation plan**.

Until then the guidance is: the advanced correlated-subquery aggregators are for
**bounded cohorts** (as-of samples, not the full entity table). Full-cohort
feature matrices should use the default-active set, or run the advanced set
through the sharded materialization path on a restricted cohort.

## Consequences

- **Positive:** correctness is fully fixed and tested now, without blocking on a
  risky performance rewrite; the scaling curve is documented, not surprising.
- **Negative:** the advanced aggregators remain impractical for full-cohort
  materialization until the follow-up plan lands; users must bound the cohort.
- **Follow-up:** a dedicated plan will benchmark the scaling curve (100 / 1k /
  10k rows), `EXPLAIN ANALYZE` the dominant aggregators, and evaluate the
  pre-aggregation vs. LATERAL-rewrite options against the
  `test_all_aggregators_execution.py` correctness harness (which guarantees any
  optimization preserves values).
