# 0010 — Set-based pre-aggregation for the correlated-subquery aggregator tier

**Status:** Accepted
**Date:** 2026-07-06
**Deciders:** Adolfo De Unánue
**Supersedes the follow-up of:** [ADR-0009](0009-correlated-subquery-aggregator-scaling.md)

## Context

ADR-0009 documented the full-cohort scaling cliff of the correlated-subquery
aggregator tier and deferred the remediation to its own plan
(`specs/correlated-subquery-aggregator-scaling.html`). That plan's Phase 1
required a measured head-to-head between the two candidate rewrites before any
planner code changed. This ADR records the outcome.

**Premise check — can a CTE ever be indexed?** No. On PostgreSQL 16.14,
`CREATE INDEX` on a `WITH` alias is a hard error (`relation … does not exist`).
Beyond that syntactic fact, featurizer's `<child>_transform` is referenced
multiple times (outer projection *and* each correlated subquery), so PostgreSQL
auto-materializes it into an *unindexed* working table; even forcing inlining
with `NOT MATERIALIZED` does not help because the transform layer carries
computed/window columns, so a correlated subquery still recomputes per group
rather than probing an index. Measured (50k rows, 500 groups, one subquery):
correlated-vs-CTE 547 ms, `MATERIALIZED` 535 ms, temp-table+index 19 ms,
set-based single scan 5.5 ms.

**Candidate A — set-based pre-aggregation.** Replace each aggregator *family*'s
per-group correlated subquery with one companion CTE: a window pre-pass
(`LAG(...) OVER (PARTITION BY child_key ORDER BY ts)`) that scans the child
stream once, wrapped by a plain `GROUP BY child_key` reduction that computes
every family member in the same pass.

**Candidate B — materialized + indexed temp table.** `CREATE TEMP TABLE … AS`
the child stream, `CREATE INDEX (child_key, …)`, and run the *unchanged*
correlated SQL against it so the per-group subquery probes the index.

Gap-family spike (6 aggregators, 1k parents × 20 children = 20k child rows,
1000 groups, ephemeral PostgreSQL 16):

| approach | wall-clock | vs status quo | correctness |
|---|--:|--:|---|
| status quo (correlated vs unindexed CTE) | 2003 ms | 1× | oracle |
| Option B (temp table + index) | 51 ms | ~39× | equal |
| **Option A (window pre-pass + GROUP BY)** | **9.8 ms** | **~205×** | **equal (0 mismatched rows)** |

Option A's per-run time is for all six gap aggregators; the status-quo and
Option B figures ran only four subqueries, so Option A's advantage is
understated. Option A reproduced the correlated oracle's values exactly (0
mismatched rows over 1000 groups; `gap_stddev` equal to 10 decimals).

## Decision

**Adopt Option A (set-based pre-aggregation).** It beats the plan's decision
rule outright — ≥10× the status quo *and* ≥2× Option B — and, being the same
complexity class as the default-active tier (which the baseline shows is flat:
0.12 s → 0.26 s from 1k → 10k parents), it removes the cliff rather than merely
softening it. Option B stays O(groups × subqueries) index probes and would carry
the temp-table lifecycle plus ADR-0006's as-of-materialization residual
(`NotImplementedError` on as-of LATERAL in a materialized synth); it is retained
only as the spike's documented control, not shipped.

The rewrite is **opt-in per aggregator** via a `preagg` protocol on
`SubqueryAggregator`: an aggregator that returns a pre-aggregation spec is routed
to a companion CTE; every other aggregator keeps today's correlated path
bit-for-bit. Output column names and manifest labels are produced by the
unchanged `_build_name`/`_build_label`, so the ADR-0007 / triage-pg naming
contract is preserved (enforced by the name-stability snapshot in every batch
gate).

**Scope.** The 27 migratable subquery aggregators (introspected count — ADR-0009's
"26" undercounted; the 9 special-config families, incl. the 4 spatial subclasses
of `SubqueryAggregator`, stay correlated). Predicate-driven, two-window drift,
and spatial families are explicit non-goals: they fire only under special config,
are excluded from the acceptance measurement, and migrate later only if a real
workload demands it.

## Consequences

- **Positive:** the advanced tier becomes practical for full-cohort
  materialization (Phase 4 acceptance target: synthetic 10k all-agg < 60 s;
  dirtyduck all-agg < 120 s, was > 300 s). Values are provably preserved (golden
  equality harness, captured pre-rewrite). No new runtime dependency.
- **Negative:** two aggs-CTE code paths coexist (plain correlated + companion
  pre-agg) until/unless the special-config families are migrated; the companion
  CTE builder is new planner surface to maintain.
- **Preserved invariants:** one companion CTE per (family, interval) keeps the
  pre-pass as a subquery in the reduction's `FROM`, so the existing
  `ShardableCTE` prefix/suffix, synth-column pruning, and `MaterializationKey`
  machinery apply unchanged (`featurizer/sharding.py` untouched).
- **Follow-up:** migrate families in gated batches (gaps → categorical → numeric
  stream → sequence), each verified against the golden values and the
  full-registry execution harness before the next begins.
