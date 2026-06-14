# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once a release is cut.

## [Unreleased]

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

### Fixed

- `#1` transform CTE re-rendered aggregate definitions against synth → reference
  by name. `#2` boundary child not materialized → depth bounds recursion only.
  `#3`/`#4`/`#5`/`#6` as-of join projection, key projection, grace-clause dialect
  safety, and PK==FK double projection. `#7` `daterange @> timestamp` invalid →
  `::date` cast at every interval window. `#8` >63-byte feature names collide
  after PostgreSQL truncation → stable hash cap (`pg_identifier`).

### Changed

- Registered-primitive counts: 69 aggregations, 83 transformers (was 43 / 71).
  Peer-group, spatial, and φ-bridge features are planner passes, not registry
  primitives.
