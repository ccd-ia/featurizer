# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once a release is cut.

## [Unreleased]

## [0.4.0] - 2026-06-21

### Added

- **Fixed-vocabulary one-hot encoding for direct (target-entity) categoricals.**
  A direct variable may now declare a `role` (`identifier` | `categorical` |
  `numeric`). A `role: categorical` variable is expanded into deterministic 0/1
  one-hot columns over a **fixed** vocabulary; a `role: identifier` variable is
  excluded from the output (loudly); `role: numeric` (and the no-role default)
  pass through as today — but a raw `text`/`categorical` direct variable left
  unencoded now emits a warning (the footgun that crashes a downstream encoder).
  Featurizer is **split-blind and fit-free**: the vocabulary is resolved from a
  declared `vocabulary: [...]` list or, failing that, the column's PostgreSQL
  `ENUM` labels — it is **never** learned by scanning the data (that fitted,
  split-sensitive transform belongs to the consumer, not to featurizer). A
  variable with neither a declared vocabulary nor an introspectable `ENUM` fails
  loud. New module `featurizer/categoricals.py`; new ADR-0007.
  - **Column-naming contract** (stable, for downstream consumers): each one-hot
    column is named `"<entity_alias>.<column>=<value>"` (e.g.
    `"facilities.facility_type=Restaurant"`), a quoted PostgreSQL identifier
    capped at 63 bytes by the existing `pg_identifier` hash-truncation. A NULL or
    out-of-vocabulary value yields an all-zero row (never a crash). The columns
    are additional numeric feature columns on the existing `query` / `to_arrow` /
    `to_parquet` / `to_dataframe` / `to_tables` paths; the consumer strips key
    columns + `*__missing` and treats the rest as features.
  - `Featurizer.__init__` gains an optional `connection=` used **only** to read
    `ENUM` labels when no `vocabulary` is declared (else one is opened from
    `DATABASE_URL` / `PG*`); a declared vocabulary keeps `query` / `--show-sql`
    fully DB-free.
- **Feature manifest.** `Featurizer.feature_manifest` (and
  `Featurizer.manifest_dataframe()`) map every output column to its full,
  untruncated intended `label` — recovering the human-readable name that the
  63-byte identifier cap erases — with a `truncated` flag, `kind`
  (`one_hot` | `variable` | `derived`), owning `entity`, and, for one-hot
  columns, the `source_column` and `value` they encode. Useful for human/partner
  labels, plot legends, and joining readable names back onto the matrix. New
  module `featurizer/manifest.py`.

## [0.3.0] - 2026-06-19

### Added

- **Temp-table materialization of oversized non-target child CTEs (issue #7).**
  Column-group sharding (0.2.0) splits the *target's* output but reuses every child
  CTE whole, so a single non-target child CTE wider than PostgreSQL's 1664-entry
  limit could not be made to fit — the cascade is inherent (an oversized child agg
  forces its consumer `synth`/`transform` over the limit too). Such a chain is now
  materialized bottom-up into keyed `TEMP`-table shards via a
  `CREATE TEMP TABLE … ON COMMIT DROP AS …` preamble run on one (non-autocommit)
  connection before the column-group queries, which are rewritten to read the
  shards. The temp tables are **`(as_of_date × entity)`-keyed feature tables** (the
  triage as-of feature-table shape): the causal `aod.as_of_date` filter, bound only
  in the outer lateral, is reintroduced via `cross join as_of_dates` once and
  carried/correlated downstream. `to_arrow` / `to_parquet` / `to_dataframe` run the
  preamble transparently; the rejoined matrix is value-identical to the
  (smaller) single query. See [ADR-0006](docs/adr/0006-temp-table-materialization.md).
- **`Featurizer.to_tables(schema)` — persist mode.** Writes the feature matrix as
  triage-style feature-group tables `"<schema>"."<stem>_group_<NNN>"` keyed on
  `(as_of_date, <target id>)`, idempotently (drop-if-exists + create), and returns a
  manifest of `FeatureGroupTable`s — the contract triage-pg consumes. The issue-#7
  intermediate shards stay ephemeral; only the final groups are persisted.
- **`Featurizer.to_dataframe` now handles wide / oversized-child configs.** A new
  one-connection `QueryExecutor.to_dataframe_materialized` runs the preamble + every
  column-group query on a single connection and re-joins them on
  `(as_of_date, <target id>)`; the fast `records` path is kept for configs that fit
  one query. `to_dataframe` gains a `connection=` kwarg (parity with `to_arrow`) so
  it can see session `TEMP` tables.
- **`Featurizer.materialization_ddl`** exposes the `CREATE TEMP TABLE` preamble for
  SQL-only callers, and **`Featurizer(..., materialize_threshold=N)`** lowers the
  1664 trigger (advanced / testing).

### Changed

- **pyarrow is now a type-check-time dev dependency.** Added to
  `[dependency-groups] dev` so `basedpyright` resolves the Arrow signatures (and is
  clean) without the runtime `[parquet]` extra — guarding the imports under
  `TYPE_CHECKING` alone was insufficient. End users still gate Arrow features behind
  `featurizer[parquet]`.
- **`warn_oversized`** now warns only for oversized intermediate CTEs that *cannot*
  be materialized (no join key — an id-less entity); materializable ones are handled
  silently.

### Known limitations

- An oversized child `synth` containing an as-of `LATERAL` join (a forward temporal
  relationship) is not yet materializable and raises `NotImplementedError` with
  guidance rather than emitting incorrect SQL. Peer-group / spatial / graph
  (`verbatim`) CTEs and id-less entities also remain `warn_oversized` bounds.

## [0.2.0] - 2026-06-17

### Added

- **Configurable as-of boundary (issue #1).** A single `featurizer/boundary.py`
  helper (`causal_predicate` / `daterange_window`) defines the point-in-time cut
  once; every graph / peer / spatial / aggregation / subquery builder routes
  through it. New top-level config key `as_of_boundary: inclusive | exclusive`
  (default `inclusive`, `<=`) selects whether an event dated exactly on the
  `as_of_date` is knowable; `exclusive` uses `<` and a half-open `[)` interval
  window. The reversed `aod.as_of_date >= temporal_ix` spelling in
  `_build_aggregations_cte` was rewritten to the canonical orientation. Default
  behavior is byte-identical.

- **Column-group sharding for wide feature matrices (issue #7).** PostgreSQL caps
  a result/CTE target list at 1664 entries, and the program's widest tuple is the
  `<target>_transform` CTE itself, so a wide config (variables × aggregations ×
  intervals × transformers) produces SQL PostgreSQL rejects. `Featurizer` now
  partitions the matrix into ordered column groups, each a self-contained query
  whose every intermediate CTE (target `transform`/`synth` and per-child `aggs`)
  is pruned to only the columns that group needs. New `Featurizer.query_groups`
  returns `OrderedDict[str, str]` (`group_<NNN>` -> SQL); every group leads with
  `(as_of_date, <target id>)` so the groups re-join into the full matrix.
  `to_arrow()` returns one `pyarrow.Table` when the config fits, else an
  `OrderedDict[str, pyarrow.Table]`; `to_parquet(path)` writes one file at `path`
  when it fits, else `path/group_<NNN>.parquet`. Null fidelity is preserved per
  group. See [ADR-0005](docs/adr/0005-column-group-sharding.md).
- **Arrow / Parquet output (`[parquet]` extra).** `Featurizer.to_arrow()` returns
  a `pyarrow.Table` and `Featurizer.to_parquet(path)` writes Parquet, both backed
  by psycopg binary `COPY (<query>) TO STDOUT (FORMAT binary)` decoded
  column-by-column into Arrow. The full result set never round-trips through
  pandas, SQL `NULL`s are preserved as Arrow nulls (not `NaN`), and `as_of_date`
  + the target id are ordinary columns (no index). Computed `numeric` aggregates
  cast to `float64` by default (`numeric_as_float=True`). `pyarrow` is a lazy,
  guarded import; the core package works without the extra. Install with
  `uv sync --extra parquet`.
- **Fit-free imputation on the Arrow path.** `impute_arrow()` mirrors
  `impute_features()` on a `pyarrow.Table` (count-like → 0, measures left null,
  stable `<feature>__missing` indicators), exposed via `impute=True` on
  `to_arrow`/`to_parquet`. The `<feature>__missing` suffix is now a documented,
  stable contract (`featurizer.MISSING_INDICATOR_SUFFIX`) shared by both paths.

### Changed

- **`Featurizer.query` refuses over-wide configs instead of emitting invalid SQL.**
  When the feature matrix exceeds PostgreSQL's 1664-entry target-list limit,
  `.query` now raises a clear `ValueError` pointing at `.query_groups` /
  `.to_parquet` / `.to_arrow` (column-group sharding) rather than returning SQL
  PostgreSQL would reject. Configs that fit are unchanged. The matrix is never
  silently truncated.
- **Whole-matrix measure imputation is gated as leakage.** `measure_strategy` in
  `{"mean","median"}` on the engine paths (`to_dataframe`/`to_arrow`/`to_parquet`)
  fits the fill over the entire returned matrix — temporal leakage (ADR-0001). It
  now requires an explicit `allow_full_matrix_fit=True` and emits a runtime
  warning even then. The standalone `impute_features` helper stays ungated for
  callers that pre-split their own data.

### Fixed

- **`ge` operator and `last_value` frame (issue #4).** `ge` rendered the invalid
  operator `=>` (now `>=`); `last`/`last_value` used the default window frame and
  silently returned the current row (now framed `rows between unbounded preceding
  and unbounded following`, returning the partition's actual last value).
- **Deterministic `Feature.short_name` (issue #5).** Long names were truncated via
  process-salted `hash()` (a different value each interpreter run); they now route
  through the deterministic `pg_identifier` scheme (`raw[:54] + "~" + md5[:8]`),
  with cross-process and collision tests.

### Tests / CI

- **Default-active primitives are executed against PostgreSQL (issue #6).**
  `tests/integration/test_default_primitives_execution.py` runs the generated SQL
  for every default-active aggregation and transformer on known fixtures and
  asserts the computed values (not just that the SQL parses), with a checklist
  test ensuring coverage. CI installs the `[parquet]` extra and runs the
  executed-SQL suite.

## [0.1.1] - 2026-06-17

### Fixed

- **Rolling ordered-set aggregates are now PostgreSQL-valid.** `rolling_median_*`
  and `rolling_iqr_*` rendered `percentile_cont(…) within group (…) OVER (…)`,
  which PostgreSQL rejects (no `OVER` on ordered-set aggregates). They now render
  as a row-framed correlated subquery over the entity's `_synth` rows (the
  transform CTE aliases its source row as `_ego` to correlate).
- **`holt_winters_trend_*` time axis.** `regr_slope(value, date)` is invalid
  (the regressor must be numeric); regress against `extract(epoch from <ts>)`.

### Changed

- **Examples execute on PostgreSQL** instead of SQLite (the engine emits
  PG-dialect SQL, so SQLite `--execute` never worked). Each example loads into a
  per-example schema via `DATABASE_URL`/`PG*`, configs select a focused primitive
  set (the full set exceeds PostgreSQL's 1664 columns-per-row limit), and
  `just example NN` / `just examples` run them over the ephemeral `db-up` harness.
  Example 04's custom primitives were rewritten to the current
  `Aggregator`/`Transformer` API. `--show-sql` remains database-free.

## [0.1.0] - 2026-06-16

### Added

- **Peer-group features (M1d)** — `peer_groups: [{by: <column>, measures: […]}]`
  on an entity. Per ego, leave-one-out and `<= as_of_date`-bounded:
  `PEER_GROUP_SIZE`, `PEER_EVENT_RATE` (per backward child), and per measure
  `PEER_MEAN` / `EGO_MINUS_PEER_MEAN` / `PEER_ZSCORE` / `PEER_PCTILE`. (ADR-0004;
  `docs/peer-group-model-alternatives.org`.)
- **Spatial second-table features (M1d)** — top-level `spatial_relationships`:
  `COLOCATION_COUNT`, `DISTANCE_TO_NEAREST`, `KDE_INTENSITY` over a second
  entity's rows within a metric radius (plain lat/lon haversine), causally
  bounded; self-relations exclude the ego.
- **φ-bridge precompute companion (M2)** — `featurizer/bridge/`: a
  `BridgeComputer` that materializes a non-SQL feature column the spine
  aggregates as a `Variable`, with a fail-fast causal guard (`assert_pre_t0`).
  Four exemplars (`MarkovSurprisalBridge`, `TfidfTopicShareBridge`,
  `PageRankBridge`, `SentenceEmbeddingBridge`) and the `[bridge]` extra.
  (ADR-0001, ADR-0003.)
- **Recursive graph families (M1b-2)** — k-hop, common-neighbours, Jaccard,
  Adamic-Adar, reciprocity, clustering over an edge-table entity, in pure SQL.
  (ADR-0002.)
- **Markov sequence aggregators (M1c)** — `recurrence_interval`,
  `markov_conditional_entropy`, `max_transition_prob`, `first_passage_time`.
- **Lexical text transformers (M1a)** — 9 Text Path-1 transformers (pure SQL).
- **Realistic integration tier** — Chicago Food Inspections and DonorsChoose
  seeders, an ephemeral Docker PostgreSQL workflow (`just db-up/seed/…`), and a
  three-tier test convention (DB-free shape guard, inline PG value test,
  realistic assertion vs an independent recomputation).
- Project artifacts: `CONTEXT.md` glossary, `docs/adr/` (0001–0004),
  `CONTRIBUTING.md`, and CI (`.github/workflows/test.yml`).
- README badges (CI, license, Python, type-checked) and an architecture diagram
  (`docs/images/architecture.svg`).

### Fixed

- `#1` transform CTE re-rendered aggregate definitions against synth → reference
  by name. `#2` boundary child not materialized → depth bounds recursion only.
  `#3`/`#4`/`#5`/`#6` as-of join projection, key projection, grace-clause dialect
  safety, and PK==FK double projection. `#7` `daterange @> timestamp` invalid →
  `::date` cast at every interval window. `#8` >63-byte feature names collide
  after PostgreSQL truncation → stable hash cap (`pg_identifier`).
- Window-function transformers (the cumulative `WindowFunctionTransformer`
  family and the ranking/`DistributionTransformer` family) dereferenced
  `parent.id.name` before the `None` check, raising `AttributeError` on an
  entity without an `id` (e.g. `id: ~`). They now no-op when there is no
  partition key — which previously crashed the documented `Featurizer(...).query`
  smoke test. Regression test added.

### Changed

- Registered-primitive counts: 69 aggregations, 83 transformers (was 43 / 71).
  Peer-group, spatial, and φ-bridge features are planner passes, not registry
  primitives.
- Type checking tightened to basedpyright **strict** (`pyrightconfig.json`). The
  `reportUnknown*` rules stay off at the untyped third-party boundaries
  (`records`, `psycopg`, `pandas`); tightening those is tracked as future work.
- Logging for agent/operator debuggability: planner `_debug` payloads now carry
  synthesized feature **names** (not just counts); `sql.render()` logs CTE count
  and query length; the executor wraps database failures in a `RuntimeError` that
  logs the full rendered SQL so a failing CTE can be traced back to its builder.
- `_haversine_m` → `haversine_m` (now public; shared by the planner's spatial
  pass).
