# 0015 — v1.0 API stability commitment

**Status:** Accepted (user sign-off 2026-07-19; frozen at the v1.0.0 tag)

**Date:** 2026-07-19

**Deciders:** Adolfo De Unánue

## Context

Through v0.9.x, "stable" was a vibe: the README claimed maturity the test
suite partly verified, but nothing said *which* surfaces a consumer may rely
on across upgrades. One consumer already depends hard on specific shapes —
triage-pg pins featurizer by git tag and joins on the ADR-0007 one-hot names
and the `to_tables` manifest. 1.0.0 exists to turn that reliance into a
written commitment with semver semantics behind it.

## Decision

### Frozen at 1.0.0 (breaking any of these requires a major version)

1. **The YAML config schema** — every documented top-level key (`target`,
   `max_depth`, `intervals`, `aggregations`, `transformations`, `entities`,
   `relationships`) including the three planner-pass blocks
   (`peer_groups`, `spatial_relationships`, `graph_relationships`) and the
   entity/relationship sub-schemas (`variables` with `type`/`role`/
   `vocabulary`/`predicates`, `temporal` blocks with `mode: as_of`/`grace`/
   `child_timestamp`). Additive keys may appear in minors; existing keys keep
   their meaning.
2. **The `Featurizer` public surface** — construction
   (`Featurizer(config_path, validate=, materialize_threshold=, debug=)`) and
   the public members: `query`, `query_groups`, `materialization_ddl`,
   `to_dataframe`, `to_arrow`, `to_parquet`, `to_tables`, `feature_manifest`,
   `manifest_dataframe`, `entities`, `relationships`. Return shapes included:
   the grouped-output contract (single table/frame when a config fits one
   group; an ordered group mapping when sharded; re-join on
   `GroupedQueries.key_columns`), and the persisted
   `<stem>_group_<NNN>` / `<stem>_manifest` table shapes.
3. **The output-naming contract** ([[0007-direct-categorical-fixed-vocabulary]]) —
   `AGG(entity.col|interval=W)` derived names, `<entity>.<col>=<value>`
   one-hots, fixed vocabularies, split-blind columns, and the 63-byte
   hash-capped identifier rule with the full label recoverable via the
   manifest.
4. **The imputation contract** — opt-in only; count-like → structural 0,
   measures stay NULL absent an explicit strategy; the
   `<feature>__missing` indicator suffix; the ADR-0001 leakage gate on
   whole-matrix fits.
5. **The φ-bridge contract** ([[0001-phi-bridge-precompute-causal-boundary]],
   [[0014-multi-column-bridge-and-temporal-snapshots]]) — `compute()`,
   `materialize()` / `materialize_nodes()` / `materialize_snapshots()` /
   `materialize_edges()` signatures and their table shapes, `emit_yaml()`
   fragment schema (`{"entity": …, "relationship": …}`), `persist=`
   semantics, `model_vintage` metadata, and the pre-t₀ assertion behaviour.

### Explicitly NOT frozen (may change in any minor)

- Planner/renderer internals: CTE names, generated SQL text, shard
  boundaries, temp-table names, plan-size heuristics and their thresholds.
- Module layout under `featurizer/primitives/` and other private modules
  (anything prefixed `_`, and imports not re-exported from package roots).
- The *set* of registered primitives may **grow** in minors; a primitive
  removal or a change to an existing primitive's emitted values is breaking.
- Log messages, warning texts, debug payloads, and documentation.

### Semver + deprecation policy

- **Major** (2.0.0): any break of a frozen surface above.
- **Minor** (1.x.0): additive features — new primitives, new config keys,
  new bridge families, performance work.
- **Patch** (1.0.x): fixes that change no contract.
- **Deprecations** warn via loguru (once per process) for **at least one
  minor release** before removal, and every deprecation is listed in the
  CHANGELOG under the release that introduces the warning.

The policy lives in CONTRIBUTING.md alongside the release procedure; the
package classifier moves to `Development Status :: 5 - Production/Stable`.

## Consequences

- triage-pg (and any external consumer) can upgrade minors without reading
  diffs: frozen surfaces cannot move, and anything moving warns for a minor
  first.
- Engine work stays free: sharding, planner passes, and SQL emission can be
  rewritten at will as long as values and names hold (the golden-value and
  snapshot suites are the enforcement).
- The freeze covers *shapes and semantics*, not wall-clock: performance
  characteristics may change in minors (they have, in every release since
  0.6.0 — see the revalidation artifacts).
- This ADR was reviewed by a human before the tag; ADR-0014 (the bridge
  contract it freezes) received the same review at the same gate.
