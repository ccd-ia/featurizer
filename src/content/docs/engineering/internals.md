---
title: "Performance internals: why it doesn't blow up"
description: >-
  How featurizer assembles its SQL — the lateral spine, CTE structure,
  set-based pre-aggregation, planner statistics, and the sharding that keeps
  PostgreSQL under its limits — with the measured numbers.
sidebar:
  order: 0
  label: Performance internals
---

A full-default config synthesizes hundreds to tens of thousands of columns.
Naively compiled, that query melts a PostgreSQL backend — early versions
proved it repeatedly, with measurements. This page tells the story of the
current design: what the emitted SQL looks like, where each cost lives, and
which decision record carries the evidence. Every number below was measured
against live databases and is archived in the
<a href="/featurizer/specs/live-db-revalidation-v080.html">v0.8.0 validation reports</a>.

## The query shape

Everything hangs off one spine:

```sql
select aod.as_of_date, t.*
from as_of_dates as aod
cross join lateral (
  with
    <child>_synth      as (…),  -- select the child's columns
    <child>_transform  as (…),  -- transformers, row-wise or windowed
    <child>_aggs_for_<target> as (…),  -- aggregations, per join key
    <companion pre-aggregation CTEs>,  -- see below
    <target>_synth     as (…),  -- join aggregates onto the target
    <target>_transform as (…)   -- target-level transformers + one-hots
  select * from <target>_transform
) as t
```

The `cross join lateral` evaluates the feature CTEs **once per as-of date**;
the `where τ ≤ aod.as_of_date` guard and per-interval `FILTER` clauses are
the [point-in-time semantics](/featurizer/concepts/phi-theory/) made visible.

## Joins: three kinds, one contract

- **Aggregates → target**: each relationship's aggregation CTE groups the
  child stream by join key and `LEFT JOIN`s onto the target's synth CTE
  (missing groups stay NULL — no data is signal).
- **As-of parents**: a `temporal: {mode: as_of}` relationship renders
  `LEFT JOIN LATERAL (… where τ ≤ t order by τ desc limit 1)` — the newest
  state at or before each as-of date, optionally bounded by `grace`.
- **Column groups → matrix**: when the output is sharded (below), every group
  leads with the full carried identifier tuple and the executor re-joins
  groups on **all** of it — a target carrying relationship keys repeats them
  per group, and joining on `(as_of_date, id)` alone would collide.

## The correlated tier, rewritten set-based

The advanced aggregations (gap statistics, entropy/HHI/Gini, sequences,
two-window drift) were originally **correlated subqueries** — re-executed per
target row: `O(rows × features)`. On real data that was the whole cost:
`EXPLAIN (ANALYZE)` on the dirtyduck database showed nine correlated subplans
at `loops=18909` accounting for essentially all of a 356.8-second run.

The fix is one idea applied family by family: **compute each family once as a
set-based companion CTE** (a windowed pre-pass with `GROUP BY` join key —
`count(*) FILTER` shared-support counts for KL divergence, per-window
`percentile_cont … FILTER` for Wasserstein), then join it in. Decision
records: [ADR-0009](/featurizer/engineering/adr/0009-correlated-subquery-aggregator-scaling/)
(the scaling analysis),
[ADR-0010](/featurizer/engineering/adr/0010-set-based-preaggregation/) (the
rewrite), [ADR-0012](/featurizer/engineering/adr/0012-set-based-two-window-drift/)
(the drift families that were deferred and then bit hardest:
**356.8s → 27.6s** on dirtyduck all-agg).

## Planner statistics: the invisible 40×

The spine table is created by *you*, usually seconds before the query runs —
so it has **no statistics**, and PostgreSQL assumes a ~2550-row default. On
donorschoose that mis-estimate picked a catastrophic join plan: one Merge
Join was 99% of a 294-second run. The executor now issues a best-effort,
savepoint-isolated `ANALYZE as_of_dates` before every query
([ADR-0013](/featurizer/engineering/adr/0013-analyze-as-of-dates/)):
**294s → 7.5s** on donorschoose, **27.6s → 7.0s** on dirtyduck — universal,
database-agnostic, and value-preserving (stats, not data).

On top of that, featurizer-owned connections get conservative
planner/memory tuning (`SET LOCAL work_mem = '64MB'`, collapse limits 20,
`geqo` deliberately ON — the aggressive variant crashed the backend
exhaustively planning a 38-way join). A caller's `connection=` is never
tuned: `SET LOCAL` would stay in force inside *their* transaction.

## Staying under PostgreSQL's limits

PostgreSQL caps a target list at **1664 entries**, and its planner has a
second, subtler cliff: planning memory is **superlinear in same-statement
window-function count** (measured: ~675 window columns plan in ~5s, ~1350
OOM-kill the backend during a plain `EXPLAIN`). Wide configs are handled by
[column-group sharding](/featurizer/engineering/adr/0005-column-group-sharding/):

- the matrix splits into self-contained group queries that re-join on the
  carried keys;
- groups are packed by **dependency lineage** (same-source columns share a
  group), so each group's CTE closure stays small — max closure went
  979 → 285 CTEs, duplicated companion executions 899 → 18;
- a **window-function budget** (500 per group) keeps every group under the
  planning cliff;
- a **pre-flight guardrail** (`warn_plan_size`) predicts pathological plans
  at render time and names the offending groups, instead of letting a run
  die minutes in with *server closed the connection unexpectedly*;
- a **heap-row-width pre-flight** on the `to_tables` path (v1.0): a heap
  tuple must fit one 8 KiB page (~8160 bytes), so a ~1,000+-column group of
  fixed-width values that SELECTs fine still fails `create table … as` with
  *row is too big*. `to_tables` estimates every group's row width (8 bytes
  per column + header + null bitmap) and re-partitions with a lower
  per-group cap when a group would exceed the ~8000-byte budget — more,
  narrower tables instead of a crash. The estimate is deliberately simple:
  text/`numeric` columns are variable-width (TOASTable), so the budget's
  headroom absorbs moderate variance rather than modeling it.

Net effect on the worst case we have: the donorschoose `wide` config
(~36.8k columns, 32 groups) went from **backend crash** to **materializing
in ~8 minutes** — and the guardrail tells you up front that you are in an
extreme regime.

## The honest numbers

Full-cohort materialization, live databases, one as-of date
(the [v0.8.0 matrix](/featurizer/specs/live-db-revalidation-v080.html)
has per-database detail):

| database | variant | v0.6.0 | v0.8.0 |
|---|---|---|---|
| dirtyduck (22k rows) | all-agg | 356.8s | **7.5s** |
| dirtyduck | wide (1,252 feats) | crash (`ln` of negative) | **63.2s** |
| chicago311 (31k rows) | all-agg | 10.1s | **6.0s** |
| chicago311 | wide (907 feats) | crash (`ln` of zero) | **49.2s** |
| donorschoose (3k rows) | all-agg | 281.1s | **7.6s** |
| donorschoose | wide (36,802 feats, 32 shards) | backend crash | **470.1s** |

Zero duplicate column names in every cell; values proven unchanged by the
golden-value gate throughout the rewrites.

## The lesson that pays rent

Every fix above was found the same way: **run `EXPLAIN (ANALYZE)` first and
read the node with the largest self-time × loops.** The plausible hypothesis
(CTE fan-out) was a multi-hour red herring twice; the measurement named the
real cause in one shot each time. When a featurizer query is slow, start
there — not at the config.
