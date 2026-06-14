# CONTEXT — Featurizer glossary

Domain and architectural terms used across the codebase, configs, and docs. One
line per term; `_Avoid_:` lists rejected synonyms to keep naming consistent.

## Terms

**Entity**: A table of things features are computed for or aggregated from
(facilities, inspections, projects). Declared with an `alias`, `table`, optional
`id` / `temporal_ix` / `spatial_ix`, and `variables`.
_Avoid_: table, dataset, node (reserve "node" for the graph sense).

**Target**: The entity the feature matrix is produced for; one row per
`(target row, as_of_date)`.
_Avoid_: subject, label entity.

**Event / event stream**: A child entity with a `temporal_ix` whose rows are
timestamped occurrences (inspections, donations) aggregated over a parent.
_Avoid_: transaction, log (domain-specific).

**As-of join / as-of semantics**: A point-in-time join (`temporal: {mode: as_of}`)
pulling the most recent parent record knowable at the target row's timestamp.
_Avoid_: latest join, temporal join (ambiguous).

**As-of date (`aod.as_of_date`)**: The cutoff for one column of the matrix; every
feature is computed from rows knowable at that date.
_Avoid_: snapshot date, t0 (use t₀ only informally).

**Causal boundary / causal bound**: The `<= aod.as_of_date` filter that keeps a
feature from reading the future. "Leakage" is its violation.
_Avoid_: time filter, cutoff filter, look-back (look-back is the `interval`).

**Interval**: An ISO-8601 window (`P6M`, `P1Y`) restricting an aggregation to a
trailing span before the as-of date.
_Avoid_: window (overloaded with SQL window functions), period.

**Cohort**: A sampled subset of target rows used in tests (a TEMP table the
config is retargeted at) so features compute only for sampled entities.
_Avoid_: sample, batch.

**Aggregation / Transformation (primitive)**: Registered SQL feature builders —
aggregations fold a child stream up to a parent; transformations rewrite a
feature within an entity. Counts: 69 / 83.
_Avoid_: function, operator.

**Peer group**: Rows of the same entity sharing a categorical `by` column;
peer-group features compare each ego to its peers, **leave-one-out** and as-of
bounded. (ADR-0004; `docs/peer-group-model-alternatives.org`.)
_Avoid_: cohort (cohort is a sampled target set), cluster.

**Leave-one-out**: A peer aggregate that excludes the ego from its own peer set
(divide by `n - 1`), so a row is never compared against itself.
_Avoid_: jackknife.

**Co-location / spatial second-table**: Features over a second entity's rows
within a metric radius of each ego (`spatial_relationships`): co-location count,
distance-to-nearest, KDE intensity.
_Avoid_: proximity join, spatial join (the latter is the SQL mechanism).

**φ-bridge (phi-bridge)**: The precompute companion (`featurizer/bridge/`) for
non-SQL features — heavy Python computes φ per row from pre-t₀ data, materializes
a column, and the SQL spine aggregates it. (ADR-0001, ADR-0003.)
_Avoid_: plugin, extension, second engine.

**Graph family**: A graph feature over an edge-table entity (degree, k-hop,
clustering, common-neighbours, Jaccard, Adamic-Adar), computed in recursive SQL.
(ADR-0002.)
_Avoid_: network feature.

**Spine / SQL spine**: The planner + renderer that emit the single PostgreSQL
query (CTEs in a lateral join against `as_of_dates`).
_Avoid_: engine, core.

## Relationships

- The **spine** computes features for the **target** across **as-of dates**,
  enforcing the **causal boundary** on every **event stream**.
- **Aggregations** fold **events** up an **as-of join**; **intervals** restrict
  the trailing span.
- **Peer-group**, **co-location**, **graph family**, and **φ-bridge** features
  are produced by dedicated planner passes / the bridge, not the primitive
  registry, but all share the same **causal boundary**.
- A **cohort** retargets the **target** (and child **events**) to TEMP tables in
  the realistic test tier.
