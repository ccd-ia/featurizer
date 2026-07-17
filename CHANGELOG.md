# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once a release is cut.

## [Unreleased]

## [0.9.0] - 2026-07-17

The text/graph feature-family release (plan:
`specs/incorporating-text-graph-feature-families.html`): the taxonomy's
`[GAP]` substrates become shipped φ-bridge families, enabled by an additive
bridge-contract extension (ADR-0014), plus one deliberate engine addition —
the native 1-hop `graph_relationships` planner pass. Trajectory / sequence /
text-induced-edge families are the 0.9.1 line.

### Added

- **Bridge contract extensions (ADR-0014, all additive)** — `MultiColumnBridge`
  (`compute() → {pk: {col: val}}`: one expensive pass emits N declared value
  columns, with per-column variable types incl. categorical);
  **temporal snapshot sequences** (`compute_snapshots` /
  `materialize_snapshots`: rebuild the model per as-of window on the pre-t₀
  slice, asserted per window, output keyed `(entity, as_of_date)` as an
  ordinary event stream — O(windows × build) by design); `materialize_nodes`
  (per-entity output for bridges whose compute keys by node); `persist=`
  (real table for orchestrated assets vs the default session-temporary);
  and `model_vintage` + `assert_model_vintage` (pretrained-model training
  cutoff as declarable, assertable metadata — `assert_pre_t0` guards fitted
  models only). The single-column contract is regression-proven byte-identical.
- **Text Path-1 bridges** (`featurizer/bridge/nlp.py`, multilingual by
  default — Spanish register, never silent English): `SentimentBridge`
  (lexicon valence, built-in es/en/xx starter lexicons, pluggable `lexicon=`),
  `ReadabilityBridge` (Fernández-Huerta / Flesch), `LanguageIdBridge`
  (stopword-profile detection, categorical output) — all three
  dependency-free — and `NERCountsBridge` (one spaCy parse → persons / orgs /
  locations / money / dates via the multi-column contract; carries
  `model_vintage`).
- **Graph bridges**: `CentralityBridge` (one networkx build → degree / in /
  out / weighted, coreness, clustering by default; betweenness, eigenvector,
  closeness **opt-in** via `include_heavy=` so configs never get silently
  slower; snapshot-aware) and `CommunityBridge` (Louvain membership as a
  categorical column + modularity; SBM/MDL-surprise deferred — graph-tool is
  not pip-installable).
- **Native 1-hop graph pass** (the one engine change): a top-level
  `graph_relationships` config block — edge table with required `timestamp`,
  optional neighbour-state entity, `measures` / `shares` defaults from
  declared variable types — generating `DEGREE(<name>)` (+ one windowed
  variant per configured interval) and `NEIGHBOUR_MEAN` / `NEIGHBOUR_SHARE`
  columns in pure SQL, bounded by **both** the edge timestamp and the
  neighbour state's `temporal_ix`. Strictly 1-hop: 2-hop aggregation (the
  canonical temporal-GNN leakage) is not offered, and validation says why.
  Validation quality matches the spatial block (required keys, entity refs,
  family/column typo suggestions).
- Docs: **bridge cookbook** page (worked example per modality, the native
  alternative, dependency matrix), ADR-0014 in the themed index, `[GAP]` →
  `shipped 0.9.0` markers in the taxonomy doc, FAQ answer updated.
- Deps: `spacy` and `python-louvain` join the `[bridge]` extra (spaCy models
  remain separate downloads); `networkx` + `python-louvain` join the dev
  group so the hand-computed graph tests execute under plain `uv sync`.
- Tests: 60 new DB-free (contract shapes, hand-computed NLP and graph
  values, SQL-shape guards for the native pass) and 10 new live-PG
  integration tests (materialize → spine handoff per family, snapshot stream
  through the spine, planted future edge *and* future neighbour state both
  excluded).

### Added — docs hub (shipped to master between 0.8.0 and this release)

- **The docs site is now a full documentation hub on Astro Starlight**
  (aligned with triage's docs stack; plan: `specs/github-pages-docs-hub.html`):
  a 10-section walkthrough tutorial (every command executed during authoring),
  the five tutorial notebooks rendered in-theme from their committed executed
  outputs (never executed in CI), a primitives reference generated from the
  live registry (count-parity tested — it cannot drift), an authored
  configuration reference, the 13 ADRs with a themed index, and the changelog.
  Python pre-build seam `site/gen.py` (uv, `docs` group) + `astro build`;
  `site/check_links.py` gates every deploy. Validation artifacts stay
  pass-through, untouched, under `/specs/`.

- **Project site on GitHub Pages** (`https://ccd-ia.github.io/featurizer/`):
  landing page, the live-DB validation artifacts (v0.6.0 / v0.8.0), and a
  `FeaturizerViz` gallery rendered from a live 177k-row × 272-feature
  dirtyduck matrix. Deployed by `.github/workflows/pages.yml` on pushes that
  touch `site/`, `specs/`, or `docs/images/`.
- README: visualization gallery (6 real plots), latest-release and docs
  badges; the exported Table of Contents block removed (GitHub renders its
  own outline).

### Fixed

- `plot_correlation_clustermap` no longer crashes on matrices containing
  constant or (near-)all-NULL features (undefined correlations made scipy's
  linkage reject the distance matrix); such features are dropped with a
  notice.

## [0.8.0] - 2026-07-12

Sharding rework: the donorschoose `wide` config (~36.8k columns) — a backend
crash in every previous snapshot — now materializes live in ~8 minutes, and
every cell of the 3-DB × 3-variant live matrix is green (all-agg is seconds
everywhere). Full refreshed artifacts: `specs/live-db-revalidation-v080/`
(+ summary page `specs/live-db-revalidation-v080.html`); decision record:
ADR-0005 amendment.

### Changed

- **Column-group sharding now clusters columns by dependency lineage.**
  `_partition_columns` buckets the target's output columns by their
  source-CTE signature before bin-packing, so same-lineage columns share a
  group and each companion pre-aggregation CTE is emitted/executed by the few
  groups that need it instead of most of them. Measured on the donorschoose
  `wide` config (27 groups, ~14.9k columns): max per-group CTE closure
  979 → 287, total closure 11,338 → 2,428, duplicated companion instances
  899 → 18, emitted SQL 29.2 MB → 17.4 MB. Group *composition* changes
  (which columns share a `<stem>_group_NNN` table); the feature manifest's
  `feature_group` column remains the supported mapping, and output column
  names are unchanged (ADR-0007).

- **Groups are additionally bounded by a window-function budget**
  (`max_window_fns_per_group`, default 500). PostgreSQL's *planning* memory
  for N same-spec window functions in one select list is superlinear with a
  hard cliff: measured live, ~675 window columns plan in ~5s while ~1,350
  OOM-killed the backend during a plain `EXPLAIN` (fresh connection; both
  halves of the same list plan fine — count, not content). The packer closes
  a group early when adding a column would exceed the budget.

  Net effect of the two partitioning changes, measured live on the
  donorschoose `wide` config (~36.8k output columns, 3,000-row cohort) that
  previously OOM-killed the backend: **materializes end-to-end in ~8 minutes**
  (32 groups, render 26.5s + execution 461.6s), max group closure 285 CTEs,
  worst per-group `EXPLAIN` well under 2s.

### Fixed

- **Sharded re-join no longer collides on carried identifier columns.** A
  target that carries relationship keys beyond its id (donorschoose's
  `schoolid` / `teacher_acctid`) repeats them in every group query;
  `to_dataframe` merged groups on `(as_of_date, id)` only, so pandas raised
  `MergeError: duplicate columns` at the third group. The materialized path
  now merges on the full `GroupedQueries.key_columns` tuple.

- **Sharded group queries no longer carry dead companion CTEs.** Per-group
  reachability now scans the *pruned* rendering of each target-level agg CTE
  instead of its full-width body, so companion pre-aggregation CTEs whose only
  consumer columns landed in other groups are no longer emitted. PostgreSQL 16
  discards unreferenced CTEs at negligible planning cost (measured), so this
  does not change plan shape — it shrinks the emitted SQL, parse time, and
  render time on wide sharded configs.

### Added

- **Pre-flight plan-size guardrail.** `ColumnGroupSharder.plan_size_report()`
  maps each column group to its live CTE-closure size, and `warn_plan_size()`
  (wired into every grouped path) logs one loud, actionable warning when any
  group's closure predicts a PostgreSQL planner blowup — the failure mode
  diagnosed on the donorschoose `wide` config, where ~1000-CTE group queries
  took 30–45s of planning each and OOM-killed the backend during a plain
  `EXPLAIN`. The warning names the worst groups and the config levers
  (transformers / intervals / entities) instead of letting the run die
  minutes later with "server closed the connection unexpectedly".

## [0.7.0] - 2026-07-10

Performance release: the two root causes found by `EXPLAIN (ANALYZE)` on the
live triage databases (correlated two-window drift → ADR-0012; no-stats
`as_of_dates` cardinality → ADR-0013) plus conservative planner tuning as an
executor default. Full-aggregator materialization on every live DB dropped from
10–357s to ~6–8s; values proven unchanged by the golden gate throughout.

### Known issues

- **The `wide` variant (all 65 aggregators × 14 transformers) on the widest
  configs can OOM the PostgreSQL backend during query *planning*.** Diagnosed
  on live donorschoose (2026-07-10): ~14.9k output columns shard into 27 group
  queries of up to ~979 CTEs / 1.8 MB SQL each; planning a single group takes
  30–45s and spikes backend memory until the kernel OOM killer fires (observed
  at a plain `EXPLAIN`, with a 3000-row cohort — data volume is irrelevant).
  Wide-everything is an extreme, atypical config; mitigation directions
  (CTE-bounded sharding, TEMP-materialized shared pre-passes, per-group
  connections) are recorded in the project TODO.

### Changed

- **Conservative PostgreSQL planner/memory tuning is now an executor default.**
  Every generated query is a wide multi-way CTE join, which starves under
  PostgreSQL's stock `work_mem` and collapse limits. The executor now issues
  `SET LOCAL work_mem = '64MB'`, `join_collapse_limit = 20`,
  `from_collapse_limit = 20` (measured ~1.4× on dirtyduck all-agg; a supporting
  lever on top of ADR-0012/0013). `geqo` deliberately stays ON — the aggressive
  variant (256MB / collapse 30 / geqo off) crashed the backend by exhaustively
  planning a 38-way join. The tuning is applied **only to connections featurizer
  opens itself**: a caller's `connection=` is never touched, because `SET LOCAL`
  would stay in force for the remainder of the caller's open transaction. On the
  records fast path the SETs share one held connection (and transaction) with
  the query; on the psycopg paths they are savepoint-isolated and best-effort,
  like the ANALYZE. New `PLANNER_TUNING` / `tuning_statements()` /
  `apply_planner_tuning()` in `featurizer.executor`; covered by
  `tests/test_executor_tuning.py`.

- **Executor ANALYZEs `as_of_dates` before running (ADR-0013).** The caller's
  freshly-created `as_of_dates` has no statistics, so PostgreSQL assumed its
  ~2550-row default and planned the lateral-join body for the wrong cardinality —
  a single Merge Join was 99% of donorschoose all-agg's runtime. The executor now
  issues a best-effort, savepoint-isolated `ANALYZE as_of_dates` on its working
  connection first, in every path (`to_dataframe`, `to_arrow`, `to_tables`).
  **donorschoose all-agg 293.6s → 7.5s, dirtyduck 27.6s → 7.0s (~40–50×)**; values
  unchanged (`ANALYZE` refreshes stats, not data — golden gate passes).
- **Two-window drift aggregators migrated to set-based pre-aggregation (ADR-0012).**
  `kl_drift` / `wasserstein_drift`, which ADR-0010 deferred as a non-goal, were the
  entire cost of full-aggregator materialization on real data: live `EXPLAIN
  (ANALYZE)` showed 9 correlated `SubPlan`s over the child stream at `loops=18909`
  (`kl_drift` firing on ordinary categorical columns × intervals, O(target×children)).
  Rewritten as companion CTEs — recent/baseline counts via `count(*) FILTER` (KL,
  no self-join) and per-window `percentile_cont … FILTER` (Wasserstein). **dirtyduck
  all-agg 356.8s → 27.6s (~13×)**, all 272 features retained, values proven identical
  by the golden-value gate (now 29 migratable aggregators / 232 frozen cases; P3M
  cases added since drift is degenerate under P1M). Output column names unchanged
  (ADR-0007). Companion-CTE budget guard 132 → 144.

- **`ln` / `log` / `sqrt` transformers are now domain-guarded (ADR-0011).** They
  render `case when x > 0 then ln(x) end` (`>= 0` for sqrt) instead of a bare
  `ln(x)`, so an out-of-domain row becomes SQL `NULL` rather than aborting the
  whole materialization with `cannot take logarithm of a negative number`. This
  hard-broke any wide/all-transformer config the moment a transformer landed on a
  signed feature (z-score, difference, deviation) — surfaced on the live-DB `wide`
  variant. Output column names/labels are unchanged (ADR-0007). New
  `DomainGuardedTransformer` base; guards covered by
  `tests/primitives/test_transformations.py`.

### Fixed

- **Companion pre-aggregation CTE name over 63 bytes emitted an invalid bare
  `~`.** A set-based companion CTE (ADR-0010) whose `<child>_<family>_<interval>_preaggs_for_<target>`
  name exceeded PostgreSQL's 63-byte identifier limit was hash-capped by
  `pg_identifier` with a `~` separator (safe only inside quotes — output columns
  are always quoted), but `_build_preagg_cte` strips the quotes to interpolate
  the name *bare*, leaving a `~` that PostgreSQL parses as an operator
  (`syntax error at or near "~"`). This hard-broke the full-aggregator config on
  any data with long categorical column names — invisible to the DB-free tests
  and surfaced only by running the integration suite against the live
  food-inspections / dirtyduck data (8 failing realistic tests). The cap
  separator is now folded to `_` for the bare CTE identifier; CTE names are
  internal-only, so the ADR-0007 output-column naming contract is untouched.
  Regression guard: `tests/test_preagg_shape.py::test_preagg_cte_name_over_63_bytes_is_a_valid_bare_identifier`.

## [0.6.0] - 2026-07-08

Set-based pre-aggregation for the correlated-subquery aggregator tier — the
performance follow-up ADR-0009 deferred. Removes the full-cohort scaling cliff
while preserving output column names (ADR-0007) and values exactly.

### Added

- **Set-based pre-aggregation path (ADR-0010).** Each of the 27 migratable
  subquery aggregators now emits one *companion CTE* — a single window (or
  grouped-join) pre-pass over the child stream reduced by a plain `GROUP BY` —
  instead of a scalar correlated subquery evaluated once per target row. Cost
  drops from `O(target_rows × subqueries × child_scan)` to one `O(N log N)` pass
  per family. Opt-in per aggregator via `SubqueryAggregator._build_preagg`; the
  companion CTE reuses the existing join / synth-pruning / sharding /
  materialization machinery unchanged.
- **Golden-value regression harness.**
  `tests/integration/test_preagg_value_equality.py` +
  `tests/fixtures/preagg_golden_values.json` freeze the v0.5.2 correlated values
  (162 cases) and assert every migrated aggregator reproduces them exactly.
  `tests/test_preagg_shape.py` adds DB-free companion-CTE shape guards. A
  `benchmarks/` package (outside the wheel) measures the scaling curve.

### Changed

- **Advanced-aggregator full-cohort materialization is now practical.** Measured
  on a synthetic 10k-parent cohort, the all-aggregator matrix went from **>300 s
  (timeout, censored)** to **2.6 s**; the worst individual families improved
  ~150–390× (`mean_deviation` 93.9 s → 0.24 s, `trimmed_mean_10` 94.6 s →
  0.27 s, `theil` 70.2 s → 0.45 s). The default-active tier is unchanged. Output
  column names and values are byte-/value-identical to v0.5.2 (proven by the
  golden harness + the ADR-0007 name-stability snapshot).
- Families migrated: gap (`gap_mean/stddev/min/max`, `gap_cv`, `burstiness`),
  categorical (`entropy`, `hhi`), numeric-stream (`gini`, `mean_deviation`,
  `theil`, `acf_1`, `variance_ratio`, `cosinor_amplitude_weekly`,
  `trimmed_mean_10`, `median_absolute_deviation`), and sequence/transition
  (`ngram_2_freq`, `ngram_3_freq`, `sequence_entropy`, `longest_streak`,
  `state_volatility`, `transition_matrix_summary`, `rework_count`,
  `recurrence_interval`, `markov_conditional_entropy`, `max_transition_prob`,
  `time_in_current_state`).

### Not migrated (intentional)

- The special-config families keep the correlated path: predicate-driven
  (`first_passage_time`, `cross_type_latency`, `right_censoring_indicator`),
  two-window drift (`kl_drift`, `wasserstein_drift`), and spatial
  (`distance_travelled`, `radius_of_gyration`, `spatial_std`, `bbox_area`). They
  fire only under special config and are out of the full-cohort scope; they
  migrate later only if a real workload demands it.

## [0.5.2] - 2026-07-06

Advanced-aggregator hardening: full-registry execution coverage (closing the
string-shape-only blind spot), plus the runtime fixes it surfaced.

### Added

- **Full-registry aggregator execution coverage.**
  `tests/integration/test_all_aggregators_execution.py` now executes *every*
  registered aggregator on real PostgreSQL over edge-case fixtures (single-row,
  constant, zero/negative, avg-zero, single-category groups; date **and**
  timestamp temporal columns). Previously only the default-active set had
  execution coverage — the advanced tier was string-shape tested only, which is
  how the v0.5.1 cluster of runtime bugs slipped through. "Every registered
  aggregator executes without error" is now a tested invariant.

### Fixed

- **`harmonic_mean` division-by-zero.** `count(x)/sum(1/x)` raised on a zero
  value (`1/0`) and on a zero denominator. Now positive-domain and guarded:
  `case when min(x) > 0 then count(x)/NULLIF(sum(1.0/NULLIF(x,0)),0) else null end`
  (NULL on the undefined non-positive domain, mirroring `geometric_mean`).
- **`mean_deviation` restored** as a correct two-pass `SubqueryAggregator`
  (`avg(abs(x - mean))` via a correlated subquery for the mean) and re-added to
  the default set — it had been removed in v0.5.1 because the single-pass form
  nested aggregates. Verified: MAD of `[1,4,9,16]` = 5.0.
- **Planner empty-CTE bug.** A single-type aggregation set over a mixed-type
  entity graph (e.g. `[entropy]` over a numeric-only child) emitted
  `select <key>, from …` — a dangling comma. The planner now skips emitting the
  aggs CTE (and its join) when an aggregation yields no features for a child.

### Removed

- **`z_score` and `min_max_scale` dropped from the registry.** They are per-row
  normalizations, not reductions — their SQL references a bare, un-grouped
  column, invalid in a `GROUP BY` aggregate. Use the `cross_entity_zscore` /
  `cross_entity_percentile` transformers instead. (v0.5.1 had excluded them from
  the default set but kept them registered; they are now fully removed.)

## [0.5.1] - 2026-07-06

Transformer-family label truncation + a cluster of never-executed advanced
aggregator bugs found by stress-testing against three live datasets, plus a
one-hot cardinality guard and CI action bumps.

### Added

- **High-cardinality one-hot warning.** Resolving a `role: categorical`
  vocabulary (declared list or introspected `ENUM`) larger than 25 values now
  logs a warning: one-hot encoding emits one sparse 0/1 column per value, which
  is wide and weak. The nudge is to declare a top-N `vocabulary:` and let the
  long tail fall into the all-zero "other". featurizer stays split-blind (it
  cannot frequency/target-encode — those are fitted, train-only transforms), so
  a warning on the declared/ENUM size is the right lever. Every value is still
  encoded (no silent data loss).

### Fixed

- **Transformer-family names now survive PostgreSQL's 63-byte identifier cap.**
  Every transformer (the base unary path plus the window / rolling / lag / EMA /
  Holt-Winters / diff / cumulative-product / cyclical / binary / population /
  CUSUM / mean-shift families) now routes its output name through
  `pg_identifier` — a deterministic hash suffix past 63 bytes — and carries a
  full untruncated `label`. Previously these names were emitted verbatim and
  *silently truncated by PostgreSQL* at runtime, so a long transformer-wrapped
  name (e.g. `ABS(patients.MEAN(visits.ABS(visits.duration_minutes)|interval=P1D))`
  at 68 bytes) risked collapsing into an ambiguous column and carried no
  intended name for the manifest. This completes the v0.5.0 manifest-label
  work, which had wired aggregations only; the manifest now maps capped
  transformer columns back to their full names and populates their lineage and
  descriptions. Short names stay byte-identical (the ADR-0007 name-stability
  contract).

- **Temporal aggregators are now type-agnostic (date *and* timestamp columns).**
  Stress-testing against three live datasets surfaced dialect bugs that only
  appear when a temporal aggregation runs on a real column: `event_rate` /
  `time_span` emitted `EXTRACT(EPOCH FROM max - min)`, invalid on a `date`
  column (`date - date` is an integer); the `gap_*` family / `burstiness` /
  `cross_type_latency` differenced raw temporal values, and `STDDEV(interval)`
  is undefined on `timestamp` columns. All now extract epoch seconds per side
  and express the result in **days** (`EXTRACT(EPOCH FROM col)/86400.0`), which
  is numeric for both types and preserves the original integer-day output on
  `date` columns. Verified executing on both a `date` and a `timestamp` fixture.

- **`geometric_mean` produced invalid SQL** — unbalanced parentheses (syntax
  error at `else`) and base-10 `log` where the geometric mean needs `ln`. Now
  `case when min(x) > 0 then exp(avg(ln(x))) else null end` (NULL on the
  undefined non-positive domain; the `ln` argument is guarded so the aggregate
  never raises before the outer guard nulls it).

- **`skewness` / `kurtosis` rewritten as pure-aggregate raw moments.** They
  referenced a bare, un-grouped column (invalid in the `GROUP BY` aggregation
  CTE) and used the `**` operator PostgreSQL lacks. Now computed from
  `avg(power(x,k))` and `var_pop(x)` — valid SQL and statistically correct
  (a normal distribution gives kurtosis 3).

### Changed

- **`z_score`, `min_max_scale`, `mean_deviation` removed from the default
  aggregation set** (still registered / requestable). The first two are per-row
  normalizations, not reductions — their SQL references a bare column that is
  invalid inside a `GROUP BY` aggregate — and are redundant with the
  `cross_entity_zscore` / `cross_entity_percentile` transformers.
  `mean_deviation` nests aggregates (`sum(abs(x - avg(x)))`), forbidden by
  PostgreSQL; it awaits a SubqueryAggregator rewrite. Removing them keeps a
  wholesale default/wide aggregation sweep valid on real schemas.

- **`in_array` removed from the default transformer set** (still registered).
  Its `__call__` requires an `an_array` argument the planner cannot supply, so
  it crashed any wholesale default/wide transform set.

## [0.5.0] - 2026-07-05

Relationship identity + manifest persistence + CI/CD. Two long-standing
relationship-topology bugs fixed (both silent until now because every shipped
config used identical key names and at most one relationship per entity pair).

### Added

- **Named relationships** (`relationships[].name`). Parallel relationships
  between one entity pair (orders as buyer AND as seller) must each declare a
  distinct `name:` — validation ERROR otherwise. The name replaces the child
  alias in aggregation feature/CTE names (`SUM(purchases.amount|interval=P1M)`,
  `purchases_aggs_for_customers`) and qualifies columns transferred by named
  forward/as-of relationships (`"purchases.score"`). Unambiguous configs need
  no `name:` and keep byte-identical feature names (ADR-0008).
- **Manifest lineage + generated descriptions.** `ManifestEntry` gains `depth`,
  `parents` (immediate parent labels), `source_alias`, `interval`, and a
  mechanically generated human `description` templated from the primitive
  documentation. Aggregation features now carry full untruncated `label`s, so
  63-byte-capped columns map back to their intended names at any nesting depth.
- **Persisted manifest table.** `to_tables(schema)` writes
  `"<schema>"."<stem>_manifest"` beside the feature-group tables — one row per
  output column including the `feature_group` it landed in (idempotent
  DROP+CREATE, parameterized inserts, caller-owned transaction).
- **CI/CD.** `test.yml` hardened (concurrency cancellation, timeouts, packaging
  gate `uv build` + `twine check`, shipped-example config validation, 70%
  fast-tier coverage floor). New `release.yml`: pushing a `vX.Y.Z` tag guards
  tag==pyproject==CHANGELOG consistency, re-verifies the tagged commit, builds
  sdist+wheel, and publishes the GitHub release with CHANGELOG notes + assets.

### Fixed

- **Relationships with differing parent/child key names rendered invalid SQL.**
  The aggregation CTE projected/grouped by the parent-side key name (absent on
  the child stream it reads) while its join referenced the child-side name the
  CTE never output; the direct-transfer CTE had the mirror-image bug. All
  builders now reference each side's own column, the parent side carries its
  join column through synth/transform, and the issue-#7 materialization key
  follows the corrected join geometry. Configs with `parent_key == child_key`
  (all previously working ones) render byte-identical SQL.
- **Parallel relationships and diamond topologies silently dropped features.**
  The traversal guard skipped every relationship after the first that reached
  an already-built entity: the second customers→orders leg vanished (5 of 10
  features) and in a diamond `a←b←d` / `a←c←d` the d-aggregations never flowed
  through c. Entities now build once while EVERY relationship is consumed, from
  a per-entity snapshot of what its transform actually projects (only true
  cycles skip). Unnamed ambiguity is a loud validation error, never a silent
  collapse.

### Changed

- `MaterializationKey.join_key` for aggregation CTEs is now the child-side key
  (the column the CTE actually carries); identical behavior for equal-key
  configs.

### Migration

- Configs declaring two or more relationships between the same entity pair now
  fail validation until each carries a distinct `name:`. Note the previous
  behavior was silently wrong (only one leg produced features), so any such
  config was already broken — now it is loudly broken with a fix suggestion.

## [0.4.2] - 2026-07-03

### Added

- **Validation warns on unknown keys in a relationship's `temporal:` block.** The parser
  only reads `mode` / `grace` / `child_timestamp`; anything else was silently ignored,
  so a misspelled key meant a silently wrong join. The validator now emits a warning
  with the exact location (`relationships[i].temporal.<key>`) and a "Did you mean?"
  suggestion (Levenshtein plus prefix match, so `child_time` suggests `child_timestamp`).

### Fixed

- **Example 02 wrote `child_time:` instead of `child_timestamp:`** in its as-of temporal
  block. The key was silently ignored; the example only behaved correctly because the
  planner's fallback picked the child entity's declared `temporal_ix` — the same column.
  Generated SQL is unchanged; the config now says what it does.

## [0.4.1] - 2026-06-21

Documentation, examples, and test-fixture follow-up to 0.4.0 (no API or behaviour
changes).

### Added

- **Example 05 — direct categoricals, output formats & imputation** (`examples/05-categoricals-output/`).
  The first DB-executing tutorial (examples 1–4 are inspection-only): a food-inspections
  scenario that shows the 0.4.0 consumer-facing features end to end — `role: categorical`
  one-hot encoding over a fixed declared vocabulary, `role: identifier` exclusion, an
  out-of-vocabulary value and a NULL (both → an all-zero one-hot row), the
  `feature_manifest`, `to_dataframe` / `to_arrow` output, and `impute=True` with
  count-vs-measure fills and `__missing` flags. Wired into `just example 05` / `just examples`.

### Changed

- **Realigned the DirtyDuck integration fixture to triage's actual schema.** The inline
  `dirtyduck` fixture in `tests/integration/test_direct_categoricals.py` now mirrors triage's
  updated raw/clean/ontology rework: the real clean-layer ENUMs (`risk_t`, `result_t`,
  `inspection_type_t`) drive the ENUM-introspected one-hot, while `facility_type` stays
  high-cardinality TEXT (excluded as an identifier) — with a fail-loud test for one-hot-ing a
  text column that has no vocabulary. Replaces the earlier invented `facility_type` ENUM.

### Fixed

- **Repaired the example tutorial notebooks.** Every `tutorial.ipynb` setup cell tried to seed
  via `exec(open("create_data.py").read())` gated on a `data.db` (SQLite) that no longer
  exists; under `exec` the script's `__file__` is undefined, so the cell raised on every run.
  The notebooks are database-free, so the seeding cell was both broken and pointless — replaced
  with a DB-free setup cell (example 04 keeps its custom-primitive registration) and re-executed.

### Documentation

- README: new "Direct categorical variables (roles & one-hot)" and "Feature manifest" sections;
  example 01 now demonstrates `role: categorical`.

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
