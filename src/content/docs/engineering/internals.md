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
<a href="/featurizer/specs/live-db-revalidation-v100.html">v1.0.0 validation reports</a>.

## The query shape

Everything hangs off one spine:

```sql
select aod.as_of_date, t.*
from as_of_dates as aod
cross join lateral (
  with
    <child>_synth      as (…),  -- select the child's rows knowable at aod.as_of_date
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

The guard sits where an entity is **read**, in `<child>_synth`, as well as in
the aggregation. The aggregation's copy alone is enough for a window that only
looks backwards, and not for one that spans its partition: `percent_rank()`
divides by the partition's size, so a row the aggregation was about to drop
had already shaped the value it kept
([ADR-0016](/featurizer/engineering/adr/0016-leak-fixes-are-not-breaking/)).
The target's own read carries the guard too when the target declares a
`temporal_ix`: a row that does not exist yet at a date is not emitted under it
([ADR-0017](/featurizer/engineering/adr/0017-an-unknowable-row-is-not-emitted/)).
A target without one is read whole.

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
exhaustively planning a 38-way join). A caller's `connection=` never gets
these: `SET LOCAL` would stay in force inside *their* transaction. The one
setting a caller's connection does see changed, and restored, is `jit`.

## JIT compilation

PostgreSQL compiles the expressions of a query whose estimated cost is over
`jit_above_cost` before it reads a row. A generated query is a target list of
hundreds of aggregate expressions, so the compile time grows with the width of
the config and not with the data: issue #53 measured 55.6 s for one query over
five rows, and 0.1 s with `jit = off`.

`benchmarks/final_matrix.py --jit-compare` then ran the nine cells of the
live-database matrix under `jit` off, on, on, off (PostgreSQL 16.14, server
defaults, 3,000 to 30,654 target rows; every run is in
`specs/jit-on-off/raw/`). Seconds, `jit = on` / `jit = off`, best of two runs
each:

| | narrow | all-agg | wide |
|---|---|---|---|
| dirtyduck | 0.84 / 0.86 | 5.16 / 4.25 | 59.39 / 7.48 |
| chicago311 | 0.21 / 0.21 | 3.25 / 2.57 | 41.72 / 17.75 |
| donorschoose | 0.18 / 0.18 | 9.04 / 8.58 | 598.70 / 436.54 |

JIT was faster in no cell (0.84 s against 0.86 s is less than the spread between
two runs of one setting), and the returned frame was identical across the four
runs in all nine.

So `to_dataframe`, `to_arrow`, `to_parquet` and `to_tables` run their statements,
the TEMP-table preamble included, with `jit = off`. This one reaches a caller's
`connection=` too, because that is where a consumer's TEMP `as_of_dates` lives:
featurizer reads the current value, issues `SET LOCAL jit = off` (a session
`SET` on an autocommit connection, where `SET LOCAL` does nothing), and puts
back the value it found. If a statement fails, the transaction is aborted and
the rollback undoes `SET LOCAL` by itself. Callers who execute `query` or
`query_groups` themselves run `set local jit = off` in the same transaction.

Every wall-clock on this page was taken before that change, with `jit = on`.

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
(per-database detail: the [v1.0.0 matrix](/featurizer/specs/live-db-revalidation-v100.html)
and the earlier [v0.8.0 snapshot](/featurizer/specs/live-db-revalidation-v080.html)):

| database | variant | v0.6.0 | v0.8.0 | v1.0.0 |
|---|---|---|---|---|
| dirtyduck (22k rows) | all-agg | 356.8s | 7.5s | **7.0s** |
| dirtyduck | wide (1,252 feats) | crash (`ln` of negative) | 63.2s | **60.2s** |
| chicago311 (31k rows) | all-agg | 10.1s | 6.0s | **5.7s** |
| chicago311 | wide (907 feats) | crash (`ln` of zero) | 49.2s | **47.5s** |
| donorschoose (3k rows) | all-agg | 281.1s | 7.6s | **8.3s** |
| donorschoose | wide (39,022 feats, 33 shards) | backend crash | 470.1s (36,802 feats) | **501.1s** |

Zero duplicate column names in every cell; values proven unchanged by the
golden-value gate throughout the rewrites. The v1.0.0 revalidation also
measured the 0.9.x families at scale for the first time — the native
`graph_relationships` pass (13,950 live chain edges × the full cohort ×
3 as-of dates: 7.8s, hand-SQL-verified) and the bridge snapshot costs
(cheap centrality tier 0.5s vs `include_heavy` 17.2s over 3 windows —
the measured reason heavy metrics are opt-in); see the
[bridge cookbook](/featurizer/engineering/bridge-cookbook/) for the
worked numbers.

## The lesson that pays rent

Every fix above was found the same way: **run `EXPLAIN (ANALYZE)` first and
read the node with the largest self-time × loops.** The plausible hypothesis
(CTE fan-out) was a multi-hour red herring twice; the measurement named the
real cause in one shot each time. When a featurizer query is slow, start
there — not at the config.
