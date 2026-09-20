# 0006 — Oversized child CTEs materialize into as-of TEMP-table feature tables

**Status:** Accepted

**Date:** 2026-06-17

**Deciders:** Adolfo De Unánue

## Context

Column-group sharding (ADR-0005) splits the *target's* output but reuses every
child CTE whole, so a single **non-target child** CTE wider than PostgreSQL's
1664-entry target-list limit could not be made to fit — the documented limitation
of ADR-0005. The cascade is inherent: an oversized deeper-chain agg
(`items_aggs_for_orders` when `orders` is not the target) forces its consumer
`orders_synth` over the limit too, because the synth projects every aggregate
column (`planner.py` records each agg column as a synth column), and then
`orders_transform` in turn. So the un-fixable unit is the whole **non-target
chain** (agg → child synth → child transform), up to but excluding the target
(whose pieces ADR-0005 already prunes per group).

A second, subtler problem only surfaced under real-PostgreSQL execution: the agg
CTEs carry a causal point-in-time filter `where <child temporal> <= aod.as_of_date`,
and `aod` (`as_of_dates as aod`) is bound **only in the final query's outer
lateral**. Lifting such a CTE into a standalone `CREATE TEMP TABLE … AS SELECT …`
loses that binding (`missing FROM-clause entry for table aod`).

## Decision

Materialize each oversized non-target child CTE into keyed **TEMP-table shards**,
bottom-up, and rewrite every downstream reference from an inline CTE into reads of
those shards. Execution becomes a session-scoped sequence: a
`CREATE TEMP TABLE … ON COMMIT DROP AS …` preamble, then the target column-group
SELECT(s), all on one non-autocommit connection (so the shards live for the
transaction and drop at close).

The temp tables are **(as_of_date × entity)-keyed feature tables** — the same
shape the triage feature-group contract uses. `MaterializationPlanner.build()`
computes as-of dependency bottom-up (a CTE is as-of-keyed when its body references
`aod.as_of_date`, or any materialized upstream is). Such a CTE is built **one
as-of date at a time**, in the single query's own shape —
`select aod.as_of_date, _m.* from as_of_dates aod cross join lateral (<the CTE's
select>) _m` — so `aod` is in scope for every inline upstream and every window
sees one date's rows. A materialized upstream it reads by name is put back under
that name inside the lateral, cut to `as_of_date = aod.as_of_date`. Shards
re-join on `(as_of_date, key)`; a synth's join to an as-of agg shard correlates
on `<shard>.as_of_date = aod.as_of_date`; and the consuming target-level agg
reads the shards through a source already cut to the current as-of date.

*Amended 2026-09-19 (issue #27).* The first form introduced `as_of_date` with a
flat `cross join as_of_dates aod`. That equals the single query only while
nothing windows: a transform over `(as_of_date × row)` shards partitions by the
entity id alone, so with two as-of dates `cum_sum` summed both copies of every
row. It also spliced the consumer's as-of cut after the first `where` in its
text, which in an aggregation with an interval belongs to a `filter (where …)`.
Both were invisible to the equality tests, which used one as-of date.

*Amended 2026-09-20 (issues #37, #52).* Until now this path had only been run
with a forced threshold on narrow configs, which proves the shape of the
preamble and nothing about the limits it exists to get under. At a real width
it met two. A shard is a heap table, and 1,400 non-null `float8` columns are an
11,232-byte row against an 8,160-byte page; shards are now sized by the same
row-width estimate `to_tables` uses for its group tables (about 980 columns).
And the shards of a materialized CTE were re-joined with `select *`, which is
over the 1664-entry target list by definition, in a FROM subquery, a CTE and a
lateral alike; every re-join now projects only the columns its reader names,
joins only the shards that carry one, and a chunk whose reads would still be
too wide is halved. `tests/integration/test_materialization_real_width.py`
runs the path at the default threshold over a 1,813-column child chain and
checks it against a narrow single query.

Each oversized CTE is already a `ShardableCTE` keyed on a single join key, so the
only new planner metadata is `materialization_keys` (cte → join key + the
consumer's join clause) — the one datum the sharder cannot recover from CTE text
without parsing column lists (which the sharding design forbids).

Public surface:

- `to_arrow` / `to_parquet` / `to_dataframe` run the preamble transparently on the
  connection they manage; output shapes are unchanged (single artifact when it
  fits, grouped/rejoined when wide). `to_dataframe` gains a `connection` kwarg
  (parity with `to_arrow`) and a one-connection `QueryExecutor.to_dataframe_materialized`.
- `Featurizer.materialization_ddl` exposes the preamble for SQL-only callers; the
  group queries from `query_groups` then presuppose it ran on the same session.
- `Featurizer(..., materialize_threshold=N)` lowers the 1664 trigger (advanced /
  testing — forces materialization on a small config without a 1664-wide CTE).

## Consequences

- A config with an oversized non-target child — rejected by PostgreSQL as a single
  query today — runs end-to-end (at a real width since the 2026-09-20
  amendment above). The rejoined materialized matrix is provably
  identical to the (smaller) single-query result, value-for-value including NULLs;
  proven in `tests/integration/test_sharding.py` for the depth-3 chain, the
  branching chain (two grandchildren), and through `to_arrow` / `to_dataframe`.
- The intermediate temp tables are exactly the triage as-of feature-table shape,
  which `Featurizer.to_tables(schema)` persists as feature-group tables
  `"<schema>"."<stem>_group_<NNN>"` keyed on `(as_of_date, id)` (idempotent
  drop+replace; the #7 intermediate shards stay ephemeral). Returns a manifest of
  `FeatureGroupTable`s — the contract triage-pg consumes. The preamble is made
  idempotent on a reused connection by a `drop … if exists` before each
  `create temp table`.
- **Residual limitations (fail loud, never silently wrong):**
  - An oversized synth containing an **as-of LATERAL** join (a forward temporal
    relationship pulling the most recent child) is **not yet** materializable —
    it raises `NotImplementedError` with guidance, rather than emitting wrong SQL.
  - Peer-group / spatial / graph CTEs are `verbatim_ctes` (no `ShardableCTE`
    column metadata), so an oversized one is never detected for materialization and
    remains a `warn_oversized` bound.
  - An oversized CTE on an **id-less entity** has no join key to re-join shards on;
    `warn_oversized` surfaces it and `MaterializationPlanner` raises if asked to
    materialize it.
- Execution is now a multi-statement, session-scoped sequence for materialized
  configs (vs. a single SELECT), on a non-autocommit connection. The `records`
  fast path is kept for configs that fit one query.
