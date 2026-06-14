# 0001 — The φ-bridge precomputes across a causal boundary

**Status:** Accepted

**Date:** 2026-06-14

**Deciders:** Adolfo De Unánue

## Context

Some feature families (NER, embeddings, graph centrality, fitted sequence
models) cannot be expressed as point-in-time-correct SQL. We needed a way to
add them without abandoning the one-database, SQL-spine design or building a
second feature engine.

## Decision

Heavy Python computes a value φ **per source row** and materializes it back into
PostgreSQL as an ordinary column; the existing SQL spine aggregates it as a
`Variable` with its normal `<= aod.as_of_date` bound. Any model that φ needs is
fit **only on rows knowable as-of the cutoff** (`<= as_of`); `assert_pre_t0`
enforces this fail-fast. The bridge does I/O when called but is a library, not a
scheduler (see [[0003-bridge-orchestration-boundary]]).

## Consequences

- No second feature engine; the spine's causal guarantee is reused unchanged.
- The leakage surface collapses to one rule (fit on pre-t₀ rows) checked in one
  place, rather than being re-derived per family.
- Per-row φ is the supported shape; genuinely per-`(entity, as_of)` snapshot
  features (e.g. PageRank over a time-sliced graph) are materialized per node and
  joined on the id — still bounded, but the consumer declares the join.
- The bridge does Python-side work outside the SQL plan, so its cost and
  reproducibility live with the orchestrator, not the planner.
