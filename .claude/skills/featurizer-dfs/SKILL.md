---
name: featurizer-dfs
description: >
  Drive Featurizer — this repo's PostgreSQL-native Deep Feature Synthesis (DFS)
  engine — to synthesize a point-in-time-correct feature matrix from a relational
  schema. Use this skill whenever the user wants to generate features with
  Featurizer, author or fix a Featurizer `config.yaml`, model an entity graph for
  automated feature engineering, set up as-of (temporal) joins, choose
  aggregation/transformation primitives and intervals, render or execute the
  generated SQL, or visualize the resulting matrix. Triggers on "featurizer",
  "deep feature synthesis", "DFS", "automate feature engineering", "write a
  featurizer config", "generate features from the entity graph", "as-of feature
  join", "feature matrix from relational data", "list-primitives", "featurizer
  config.yaml". This is the AUTOMATED-breadth feature layer; for hand-crafted
  bespoke SQL features and the cohort/label/temporal-CV scaffolding, use
  `pg-ml-pipeline` instead — the two compose. Do NOT use for model training
  (`ml-implement`), evaluation (`ml-evaluate`), or problem formalization
  (`ml-formalization`).
user-invocable: true
---

<!-- CANONICAL COPY — ships with the engine and is pinned to it by
     tests/test_skill_parity.py (version, primitive counts, curated defaults,
     every `f.<attr>` it names). Update the body as part of every release, next
     to the CHANGELOG section (CONTRIBUTING.md → Release process). Vendored
     snapshots of the BODY (from the H1 down) live in ~/.claude/skills/ and in
     the claude-tips skills catalog; they keep their own frontmatter. -->

# Featurizer — Deep Feature Synthesis (PostgreSQL) — v1.1.1

Featurizer implements Deep Feature Synthesis for relational PostgreSQL data with
first-class temporal semantics. You declare an entity graph once; it traverses
relationships, applies aggregations across them, and emits **pure SQL** that
computes a point-in-time-correct feature matrix indexed by `(as_of_date,
<entity_id>)`. No data movement — features are computed where the data lives.

The Python API is trivial (`Featurizer("config.yaml").query` → SQL string;
`.to_dataframe()` → matrix). **All the difficulty is in `config.yaml`**, which is
convention-heavy and leakage-prone. This skill's job is to get that config right.

Since 1.0.0 the public surface is **frozen** (ADR-0015): config schema, the
`Featurizer` methods and their return shapes, output column naming, the
imputation contract and the φ-bridge contract only change with a major version
and a deprecation cycle. See "Stability & support" below for what is *not*
frozen.

## When Featurizer vs `pg-ml-pipeline`

| Use Featurizer (this skill) | Use `pg-ml-pipeline` |
|---|---|
| Automated *breadth*: hundreds of features from a clean entity graph | Hand-crafted *bespoke* domain features |
| You have well-modeled entities + relationships + event timestamps | You need the cohort, label, temporal-split, governance scaffolding |
| "Throw the kitchen sink, then prune" | "I know exactly the 12 features that matter" |

They **compose**: Featurizer's matrix joins onto `pg-ml-pipeline`'s cohort/label
tables, then feeds `ml-implement`. Featurizer does **not** build cohorts, labels,
temporal splits, or train models — keep those in `pg-ml-pipeline` / `ml-implement`.

## Mental model: entity graph + temporal semantics

- **Entities** are tables: either a *dimension/target* (one row per id) or an
  *event source* (many timestamped rows, often `id: ~` because it has no own key).
- **Relationships** connect a `parent` (the "one" side / lookup) to a `child`
  (the "many" side / event source). Direction is the #1 thing LLMs get wrong:
  - **Aggregation flows child → parent.** Many child rows roll up to one parent
    via the primitives (`SUM`, `COUNT`, `recency`, …). If your `target` is the
    parent, its event children get aggregated onto it.
  - **As-of lookup flows parent → child.** A `temporal: {mode: as_of}` block lets
    the child pull the *latest parent row as of the child's timestamp*. The
    lookup/dimension table is declared `parent`, the timestamped target is
    `child`. (In this repo's `featurizer/featurizer.yaml`, `care_plans` is the
    *parent* of `patients` with `mode: as_of, grace: P21D` — counterintuitive but
    correct: each patient row pulls its most-recent care plan.)

## Authoring `config.yaml`

**Required top-level keys:** `target`, `max_depth`, `intervals`, `entities`
(validation rejects a config missing any). `relationships` is required whenever
`max_depth > 1`.

```yaml
target: patients          # the entity the feature matrix is indexed by
max_depth: 2              # traversal depth — see "feature-count budget" below
intervals: [P7D, P1M]    # ISO-8601 durations; global default windows

# Primitive selection (optional). These TOP-LEVEL lists are the real selection
# mechanism. Three spellings, three meanings (the last one since 1.0.1):
#   omitted / null  → curated active set (count, mean, sum, stddev, min, max,
#                     median, nunique, recency, tenure + 17 default transformers)
#   [a, b, c]       → exactly those primitives
#   []              → suppress that layer: `aggregations: []` builds zero
#                     aggregation features; `transformations: []` passes
#                     features through unchanged (identical to `[identity]`).
#                     Before 1.0.1 an explicit [] silently applied the defaults.
aggregations:    [sum, mean, recency, gap_cv, entropy]
transformations: [identity, lag_1, rolling_mean_7]

entities:
  - alias: patients              # referenced by relationships
    id: patient                  # primary key column; `~` for keyless event tables
    table: semantic.patients     # schema-qualified source table
    temporal_ix: registered_at   # event-time column — drives ALL interval windows
    variables:
      gender: { type: categorical }
  - alias: measurements
    id: ~                        # event source: no own id
    table: semantic.measurements_view
    temporal_ix: measured_at
    variables:
      peso:
        type: numeric
        intervals: [P1D, P3D, P1W]   # per-variable windows override the global

relationships:
  - parent: { entity: patients,   key: patient }   # one side
    child:  { entity: measurements, key: patient }  # many side → aggregated up
  - parent: { entity: care_plans, key: patient }   # as-of lookup
    child:  { entity: patients,   key: patient }
    temporal: { mode: as_of, grace: P21D }          # latest care_plan as of patient ts
```

**The gotchas, in priority order:**

1. **`temporal_ix` is the event-time column.** Wrong column → wrong windows →
   silent leakage. An as-of join validates only if `temporal_ix` is set on *both*
   sides of the relationship (else it falls back to a static key join).
2. **Parent/child direction** — see the mental model. Aggregation = child→parent;
   as-of lookup = parent→child.
3. **`grace`** (e.g. `P21D`) on an as-of join widens how stale a parent row may be
   and still match. Set it to the real domain validity window; default is exact.
4. **`child_timestamp`** overrides the source entity's timestamp column for a
   relationship when it differs from the entity's declared `temporal_ix`. It is
   the only override key the `temporal:` block accepts besides `mode`/`grace` —
   since v0.4.2 the validator warns on unknown keys (e.g. `child_time`).
5. **`variables[*].type`** ∈ {`numeric`, `categorical`, `text`, `boolean`,
   `date`, `timestamp`, `index`, `vector`}. The first three carry the
   primitives (entropy/hhi need categorical, lexical transforms need text);
   `index` marks a join key that must be projected through an as-of join
   without being featurized. A target-entity variable can additionally set
   **`role`**: `categorical` → one-hot encoded against a **fixed vocabulary** —
   declared `vocabulary: [...]` or read from the column's PostgreSQL ENUM
   (introspected once at construction, so pass `Featurizer(..., connection=conn)`
   or have `PG*`/`DATABASE_URL` set), never learned from the data (split-blind;
   NULL/out-of-vocabulary → all-zero row); `identifier` → loudly excluded from
   the output (names, license numbers). One-hot columns are named
   `"<entity>.<col>=<value>"`. See `examples/05-categoricals-output/config.yaml`.
6. **Per-variable `intervals`** override the global `intervals` for that column.
7. **Parallel relationships need `name:`** (v0.5.0). Two relationships between
   the same entity pair (orders as buyer AND as seller) must each declare a
   distinct `name:` — validation errors otherwise (before v0.5.0 the second leg
   silently vanished). The name becomes the feature alias
   (`SUM(purchases.amount|…)`) and qualifies transferred columns
   (`"purchases.score"`). Unambiguous configs need no `name:` and keep their
   existing feature names. Parent/child key column names may differ per side —
   the engine references each side's own column.
8. **`as_of_boundary`** (top-level, optional): `inclusive` (default — an event
   dated exactly on the as-of date is knowable, `<=`) or `exclusive` (`<`).
9. **Column names longer than 63 bytes are hash-truncated** to
   `<head 54>~<8 hex>` (PostgreSQL's identifier limit; ADR-0007, frozen). Nested
   features truncate routinely — on the sample config 1,217 of 2,104 columns
   do. **Never parse or glob physical column names.** The manifest carries the
   full `label`, a `truncated` flag and lineage for every column
   (`f.feature_manifest` / `f.manifest_dataframe()`; `to_tables` persists it
   as `"<schema>"."<stem>_manifest"`), and since 1.1.0 the selection helpers
   glob that label for you:
   ```python
   cols = f.columns_matching("*frecuencia_cardiaca*")   # physical names, output order
   rows = f.manifest_matching("*(inspections.kw_*")     # full ManifestEntry rows
   ```
   Zero matches **raise `LookupError`** with near-miss suggestions (a silently
   empty feature group is the failure this exists to prevent); pass
   `allow_empty=True` when probing for optional features. Matching is
   case-sensitive and against `label` only. For the SQL side,
   `featurizer.manifest.glob_to_like(pattern)` returns a `LIKE` pattern plus its
   escape character — `_` is literal in a glob but a wildcard in `LIKE`, and
   ~95% of labels contain one.

## Beyond the registry: planner passes & φ-bridges

Some feature families are **not** registry primitives:

- **Planner passes**, driven by their own config blocks:
  - `peer_groups` (per-entity): leave-one-out comparisons against peers
    sharing a categorical — `PEER_GROUP_SIZE`, `PEER_MEAN`/`PEER_ZSCORE`/
    `PEER_PCTILE`/`EGO_MINUS_PEER_MEAN`, `PEER_EVENT_RATE`.
  - `spatial_relationships` (top-level): co-location count / distance-to-
    nearest / KDE intensity between entities with lat/lon `spatial_ix`.
  - `graph_relationships` (top-level, v0.9.0): native 1-hop graph features
    over an edge table, pure SQL — `edges: {table, source, target, timestamp}`
    (**timestamp required** — as-of by construction), optional `right`
    neighbour-state entity, `measures`/`shares` defaulting from declared
    variable types, `features ⊆ {degree, neighbour_mean, neighbour_share}`.
    Emits `DEGREE(<name>)` (+ one windowed variant per interval) and
    `NEIGHBOUR_MEAN`/`NEIGHBOUR_SHARE`, bounded by BOTH the edge timestamp
    and the neighbour state's `temporal_ix`. Strictly 1-hop (2-hop leaks
    neighbours' future labels — deliberately not offered).
- **φ-bridges** (`featurizer.bridge`, heavy deps behind the `[bridge]`
  extra): precompute → materialize a column → `emit_yaml()` a config
  fragment the spine aggregates like any `Variable`. Shipped families:
  sentiment / readability / language-id (dependency-free, Spanish-register
  defaults), spaCy NER counts, TF-IDF topic shares, sentence embeddings →
  pgvector, embedding-trajectory novelty/drift/volatility, PageRank and
  multi-metric centralities + Louvain community (rebuilt per as-of window via
  `materialize_snapshots` — never slice a full-history graph),
  change-point/periodicity, Markov surprisal, and text-induced **edge
  builders** (near-duplicate MinHash/LSH, co-mentions) whose `(src, dst, ts)`
  output feeds the graph features. Fitted models train on pre-t₀ rows only
  (`assert_pre_t0`); pretrained models declare `model_vintage`.
  `persist=True` writes real tables for Dagster/Snakemake assets. Worked
  examples: `examples/06-graph-text-bridge/` and the bridge cookbook
  (https://ccd-ia.github.io/featurizer/engineering/bridge-cookbook/).

## Feature-count budget (the explosion governor)

Feature count ≈ `entities × variables × primitives × intervals`, compounded by
`max_depth` traversal. There are **67 aggregations + 83 transformers (150
primitives)** registered — requesting all of them at `max_depth: 3` across several
intervals yields *thousands* of columns and an unusable matrix.

- Start from the curated active set (omit `aggregations:`/`transformations:`).
- Add primitives deliberately; run `python -m featurizer list-primitives -c` to
  see them grouped by category before choosing.
- Treat `max_depth` as expensive: depth 2 is the common case; depth 3 (the retail
  example) is a deliberate choice, not a default.
- Keep `intervals` to the windows the model actually needs (e.g. `[P7D, P1M]`),
  not one of every duration.

## Workflow

```bash
# 1. Validate the config BEFORE touching the DB (catches schema/key errors offline)
uv run python -m featurizer validate config.yaml

# 2. Inspect the generated SQL — read it; confirm the as-of joins are LATERAL and
#    every window filters occurred_at < as_of_date (no future leakage)
uv run python -c 'from featurizer import Featurizer; print(Featurizer("config.yaml").query)'

# 3. Discover primitives when choosing
uv run python -m featurizer list-primitives --type all --category   # add -s for example SQL
```

```python
from featurizer import Featurizer

f = Featurizer("config.yaml")
print(f.query)                    # the pure-SQL feature query
df = f.to_dataframe()             # execute against the live DB → pandas matrix
tbl = f.to_arrow(connection=conn, impute=True)  # binary COPY → pyarrow, no pandas hop
f.to_parquet("out.parquet", connection=conn)    # Arrow path written to Parquet
f.to_tables("features")           # persist triage-style feature-group tables

# select features by their full label, never by physical (possibly truncated) name
cardiac = f.columns_matching("*frecuencia_cardiaca*")
```

- **Wide matrices auto-shard into column groups** (PostgreSQL's 1664
  columns-per-row limit): `to_arrow` returns an `OrderedDict` of group tables,
  `to_parquet` writes one file per group under a directory, and `to_tables`
  persists `"<schema>"."<target>_group_<NNN>"` tables — all groups re-join on
  `(as_of_date, <target id>)` to reconstruct the full matrix. `to_tables`
  additionally pre-flights each group's heap row width (a CTAS row must fit one
  8 KiB page) and re-partitions into more, narrower tables when it would not
  (1.0.0) — so it can emit more groups than `to_arrow` for the same config.
  Documented residual: an as-of relationship *inside* an oversized child chain
  that gets materialized raises `NotImplementedError`.
- **Imputation is opt-in and fit-free by default** (`impute=False`): when
  enabled, count-like features → 0 and measures stay NULL with stable
  `<feature>__missing` indicators; a fitted `measure_strategy`
  (`"mean"`/`"median"`) additionally requires `allow_full_matrix_fit=True`
  (ADR-0001 leakage gate).
- **Performance is handled for you** (v0.6–0.8): set-based pre-aggregation
  replaced correlated subqueries, the executor `ANALYZE`s `as_of_dates` and
  applies planner tuning automatically, and wide configs shard by lineage
  with a window-function budget. If a run is slow, `EXPLAIN` first — do not
  hand-shard.

- **Validate → read SQL → execute.** Never run `.to_dataframe()` against a large
  DB before reading `.query` once — depth/interval mistakes are obvious in the SQL
  and expensive in the executor.
- Debug traversal with `FEATURIZER_DEBUG=1` (or `Featurizer(..., debug=True)`):
  it mirrors traversal depth, aggregation counts, and transformation totals.

## Leakage discipline (the whole point of DFS-with-time)

Point-in-time correctness is *why* Featurizer exists over a naive join. Guard it:

- Every aggregation window must reference the row's `as_of_date` and only look
  **backward**. Confirm in the rendered SQL.
- As-of joins must materialize as `LEFT JOIN LATERAL` pulling exactly one
  (nearest, within `grace`) parent row — verify, don't assume.
- The matrix is `(as_of_date, <entity_id>)`-indexed. **NULLs are signal**
  (no events in window), not errors — do not blanket-impute upstream.

## Stability & support (v1.0+)

- **Frozen (ADR-0015):** the YAML config schema including the
  `peer_groups` / `spatial_relationships` / `graph_relationships` blocks, the
  `Featurizer` public surface and return shapes, the ADR-0007 output-naming
  contract (including 63-byte capping), the opt-in imputation contract, and
  the φ-bridge contract. Breaking any of these needs a major version and a
  ≥-one-minor deprecation cycle (loguru warning first).
- **Not frozen:** planner/renderer internals, CTE names, SQL text, module
  layout. Consumer code that asserts on CTE names or SQL fragments is
  asserting on internals — key off the manifest and the output columns.
- **Tested matrix:** DB-free tier on Python 3.10–3.13; the generated SQL
  executes on PostgreSQL 14/16/17 in CI. Downstream pins (triage-pg) pin the
  engine version into artifact identity, so a pin bump intentionally
  invalidates feature caches even when no column is renamed.

## Integration with the ML catalog

Featurizer produces the **feature matrix**; the rest of the problem lives in the
existing skills:

1. `ml-formalization` / `pg-ml-pipeline` define entity, cohort, label, as-of dates,
   temporal splits. Featurizer's `target` + `as_of_dates` must match the cohort's.
2. Featurizer emits the `(as_of_date, entity_id)` matrix → join onto the
   cohort/label tables from `pg-ml-pipeline` on the same key + date.
3. Hand the joined matrix to `ml-implement` for training; evaluate with
   `ml-evaluate` (and the survival/forecasting auditions there if the label is
   time-to-event or a level forecast).

## Diagnostics: `FeaturizerViz`

After materializing, `FeaturizerViz.from_featurizer(f)` gives feature-quality
plots (optional extra: `uv sync --extra viz`): `feature_summary_table`,
`plot_missing_heatmap`, `plot_correlation_clustermap` (redundancy),
`plot_feature_importance(target_col=...)`, entity embeddings, per-entity time
series. Use it to prune the explosion *after* synthesis — drop high-missing,
high-redundancy, low-importance features before training.

## Anti-patterns

- ❌ Requesting all 150 primitives "to be safe" → thousands of useless columns.
  Start curated, add deliberately.
- ❌ Inverting parent/child → empty or wrong features. Aggregation is child→parent.
- ❌ Missing/incorrect `temporal_ix` → leakage or no windowed features.
- ❌ Regressing/aggregating without confirming the SQL only looks backward.
- ❌ Executing on a large DB before validating + reading the rendered SQL.
- ❌ Selecting features by globbing *physical* column names — truncated columns
  silently fall out of the group. Glob the manifest label with
  `f.columns_matching(...)`.
- ❌ Writing `aggregations: []` on an engine older than 1.0.1 expecting "none" —
  it applied the defaults. Pin ≥ 1.0.1 or spell the layer explicitly.
- ❌ Rebuilding cohorts/labels/splits here — that's `pg-ml-pipeline`'s job.
- ❌ Imputing NULLs at synthesis time — they carry "no activity" signal.

## Quick reference

| Action | Command |
|---|---|
| Validate config | `uv run python -m featurizer validate config.yaml` |
| List primitives (by category) | `uv run python -m featurizer list-primitives -c` |
| List with example SQL | `uv run python -m featurizer list-primitives -s` |
| Render SQL | `Featurizer("config.yaml").query` |
| Execute → DataFrame | `Featurizer("config.yaml").to_dataframe()` |
| Execute → Arrow (no pandas hop) | `f.to_arrow(connection=conn, impute=True)` |
| Write Parquet (file or group dir) | `f.to_parquet(path, connection=conn)` |
| Persist feature-group tables | `f.to_tables("schema")` |
| Column ↔ label manifest (+lineage/descriptions) | `f.feature_manifest` / `f.manifest_dataframe()` |
| Select columns by label glob (1.1.0) | `f.columns_matching("*pattern*")` → physical names; `f.manifest_matching(...)` → rows |
| Same glob on the persisted manifest table | `featurizer.manifest.glob_to_like(pattern)` → `(like_pattern, escape_char)` |
| Persisted manifest table | written by `to_tables` as `"<schema>"."<stem>_manifest"` |
| Debug traversal | `FEATURIZER_DEBUG=1 …` or `Featurizer(..., debug=True)` |
| Register a primitive | `register_aggregation("name", obj)` / `register_transformer(...)` |

Core modules for bespoke workflows: `featurizer/planner.py` (traversal +
planner passes), `featurizer/sql.py` (rendering), `featurizer/executor.py`
(execution), `featurizer/validation.py` (config checks), `featurizer/manifest.py`
(labels, lineage, glob helpers), `featurizer/bridge/` (φ-bridge families).

Docs hub: https://ccd-ia.github.io/featurizer/ (configuration reference,
primitives explorer, bridge cookbook, FAQ, ADRs). Distribution is GitHub
releases on `ccd-ia/featurizer` (no PyPI, deliberate); each release also
attaches a `docs-site-vX.Y.Z.tar.gz` snapshot. Pin
`featurizer[parquet] @ git+https://github.com/ccd-ia/featurizer@v1.1.1`
(extras: `parquet` for Arrow/Parquet output, `bridge` for φ-bridges, `viz` for
`FeaturizerViz`).
