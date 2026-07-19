# 0014 — Multi-column bridges and temporal snapshot sequences

**Status:** Accepted (human-reviewed at the v1.0 freeze gate, 2026-07-19 — frozen by [[0015-v1-api-stability-commitment]])

**Date:** 2026-07-17

**Deciders:** Adolfo De Unánue

## Context

The `BridgeComputer` contract (see
[[0001-phi-bridge-precompute-causal-boundary]]) ships one φ per bridge:
`compute() → {pk: scalar|vector}` and a single pre-`fit_before` model fit.
Two whole feature families are blocked by that shape, before any family code
is written:

1. **Multi-metric families.** NER counts (persons/orgs/locations/…) and graph
   centralities (degree/coreness/clustering/…) produce many values from *one*
   expensive pass (one spaCy parse, one graph build). One bridge per metric
   re-runs the pass per column.
2. **Non-local temporal features.** Centrality is non-local — one future edge
   changes every node's score — so a backtest cohort with many as-of dates
   needs the graph rebuilt *per window* from strictly pre-t₀ edges. The
   contract has a single `fit_before`; there is no snapshot-sequence
   mechanism, and slicing one full-history graph leaks the future.

Two smaller gaps ride along: `materialize()` only writes
`create temp … on commit drop` (no way to persist a bridge output as a real
Dagster/Snakemake asset, the [[0003-bridge-orchestration-boundary]] wiring),
and `assert_pre_t0` guards **fitted** models only — a *pretrained* model
snapshot (spaCy NER, sentence-transformers) trained on post-t₀ data is silent
leakage the harness cannot see.

## Decision

Extend the contract **additively**; the single-column/vector path stays
byte-identical (regression-tested).

- **`MultiColumnBridge`** subclass: `compute() → {pk: {col: val}}` with
  declared `value_cols`; `materialize()` builds DDL/INSERT for N value
  columns, `emit_yaml()` declares one `Variable` per column.
- **Temporal snapshot sequences**: `materialize_snapshots(as_of_dates=…)`
  rebuilds the model/graph per window on the `causal_col <= as_of` slice
  (asserted per window via `assert_pre_t0`) and emits rows keyed
  `(entity, as_of_date)` — an ordinary event stream (`as_of_date` is the
  `temporal_ix`), so the spine trends centrality like any other metric. Cost
  is O(windows × build) by design; no snapshot-binning approximation.
- **`persist=` option** on materialization: default stays today's
  `create temp … on commit drop`; `persist=True` writes a real
  `create table` for orchestrated assets.
- **`model_vintage`** optional attribute (training-cutoff date) on
  `BridgeComputer`. Model-bearing bridges declare it and pin the model
  version; `None` means "unknown vintage". A strict backtest can assert
  `model_vintage <= as_of` (opt-in). This is metadata + documentation, not an
  automatic guard: `assert_pre_t0` still covers *fitted* models only.

## Consequences

- One pass, many columns: NER and centrality families become one bridge each
  instead of one per metric.
- Per-`(entity, as_of_date)` snapshot output is the causal-correct form for
  non-local graph features; the consumer pays O(windows × build) and the
  docstrings say so loudly. Cheap centralities are the default tier;
  expensive ones (betweenness, eigenvector, closeness) are opt-in.
- Pretrained-model leakage remains *possible* — the vintage metadata makes it
  visible and assertable, not impossible. This is the honest boundary of the
  harness.
- The base contract is unchanged for existing bridges; downstream configs and
  the ADR-0007 naming contract are untouched.
