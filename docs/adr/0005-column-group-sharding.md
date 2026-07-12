# 0005 — Wide feature matrices shard into joinable column groups

**Status:** Accepted (amended 2026-07-11: lineage-aware packing + window budget)

**Date:** 2026-06-17

**Deciders:** Adolfo De Unánue

## Context

PostgreSQL caps a result/CTE target list at **1664 entries**
(`target lists can have at most 1664 entries`) and a table at 1600 columns. A
wide config (child variables × aggregations × intervals × transformers) easily
produces more than 1664 features, and the program's widest tuple is the
`<target>_transform` CTE itself — so the limit bites at the *CTE boundary*, not
just the outer `select`. Sharding only the final `select *` would still emit a
`<target>_transform` tuple PostgreSQL rejects.

Three options were weighed: (A) cap/drop features to fit (silent data loss —
rejected outright); (B) one row-wise query streaming a long-format
`(entity, feature_name, value)` table (loses per-feature typing and Arrow null
fidelity, and is awkward to re-pivot); (C) partition the *columns* into ordered
groups, each a self-contained query, all joinable on `(as_of_date, id)` — the
triage-pg feature-group pattern.

## Decision

Adopt **Option C: column-group sharding.** The planner records structured
metadata (`ShardableCTE` / `ColumnSpec` / `synth_column_source`) for the three
CTEs whose width can exceed the limit — the target's `transform`, `synth`, and
per-child `aggs`. A `ColumnGroupSharder` partitions the target transform columns
into ordered groups of ≤ `DEFAULT_MAX_COLUMNS_PER_GROUP` (1400, headroom under
1600), and for **each group** rebuilds a valid query in which no intermediate
CTE tuple exceeds the limit either:

- the target `transform` CTE projects only that group's feature columns
  (+ the carried identifier columns);
- the target `synth` CTE is pruned to the synth columns those transform columns
  read (tracked per column via a literal-name scan of each transform
  projection's SQL against the known synth column names), dropping the joins and
  upstream CTEs nothing in the group needs;
- each per-child `aggs` CTE is pruned to the surviving synth columns it feeds;
- every other (bounded) CTE is reused **verbatim**, pulled in by a transitive
  CTE-name reachability scan so each group's `with` list is exactly what it
  references.

Every group leads with `(as_of_date, <target id>)`, so the groups re-join into
the full matrix. Public API:

- `Featurizer.query_groups -> OrderedDict[str, str]` (`group_<NNN>` -> SQL); a
  config that fits returns a single `group_000` equal to `.query`.
- `Featurizer.query` raises a clear error pointing at the sharded API when the
  matrix exceeds the limit — it never silently truncates.
- `to_arrow()` returns one `pyarrow.Table` when it fits, else an
  `OrderedDict[str, pyarrow.Table]`; `to_parquet(path)` writes one file at `path`
  when it fits, else `path/group_<NNN>.parquet`. #2/#3 null fidelity (Arrow
  nulls, never NaN) is preserved per group.

## Consequences

- No feature is ever dropped; the failure mode for an over-wide config is a
  loud, actionable error or a set of valid grouped artifacts, not a silently
  truncated matrix or a query PostgreSQL rejects.
- The dependency tracing is a literal-name scan (transformer SQL references its
  inputs by the exact synth column name), so it is robust to how each
  transformer is implemented — no reliance on the inconsistent `Feature.parents`
  attribute.
- **Documented limitation (not silent):** sharding splits the *target's* output
  and reuses every child CTE whole, so a single *child* entity whose own
  `transform`/`synth`/`aggs` tuple already exceeds 1664 cannot be fixed by
  grouping the target — pruning operates per target column group, but a child
  CTE is shared across groups. `query_groups` / the Arrow path log a clear bound
  (`warn_oversized`) naming the offending CTE and its width; the remedy is to
  narrow that child's primitive/interval breadth or raise its relationship to
  the target. A pathological transformer fan-out that would push a *pruned*
  synth tuple over the limit fails fast with context rather than emitting
  invalid SQL.
- The grouped output is the feature-group contract other tools (triage-pg)
  already consume: independent column shards keyed on `(as_of_date, id)`.

## Amendment (2026-07-11): how columns are packed into groups

Emission-order chunking made each group's CTE closure span most of the plan
(donorschoose `wide`: closures up to 979 CTEs, 30–45s of PostgreSQL *planning*
per group, backend OOM-killed during a plain `EXPLAIN`). Packing now:

1. **clusters columns by dependency lineage** (source-CTE signature) before
   filling groups — max closure 979 → 285, duplicated companion executions
   899 → 18 instances, emitted SQL 29.2 MB → 17.4 MB; and
2. **bounds window functions per group** (`max_window_fns_per_group`, 500) —
   PostgreSQL planning memory is superlinear in same-statement window-function
   count (measured cliff: ~675 plan in ~5s, ~1350 OOM the backend; count, not
   content).

Group *composition* is therefore lineage-dependent, not positional; the
`<stem>_manifest`'s `feature_group` column remains the supported mapping from
column to group. Validated live: the donorschoose `wide` config (~36.8k
columns, previously a backend crash) materializes in ~8 minutes across 32
groups. The re-join key is the full carried identifier tuple
(`GroupedQueries.key_columns`), not just `(as_of_date, id)` — targets that
carry relationship keys repeat them per group.
