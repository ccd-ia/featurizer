# 0002 — Graph features use recursive SQL, not Apache AGE

**Status:** Accepted

**Date:** 2026-06-09

**Deciders:** Adolfo De Unánue

## Context

Graph feature families (degree, k-hop, clustering, common-neighbours, Jaccard,
Adamic-Adar) traverse an edge table. Apache AGE (Cypher-in-Postgres) is the
project-wide default for relationship-heavy data, so it was the obvious candidate.

## Decision

Compute graph features with plain recursive/standard SQL CTEs over the edge
table, **not** Apache AGE.

## Consequences

- No extension dependency: the generated SQL runs on any stock PostgreSQL,
  including the ephemeral `postgres:16` test container.
- The edge causal bound (`timestamp <= aod.as_of_date`) composes with the rest
  of the spine identically to every other CTE — one leakage idiom, not two.
- For the bounded, well-known graph families we ship, hand-written SQL is
  adequate; AGE would add a deployment surface for no gain here. If a future
  family needs deep variable-length traversal that recursive CTEs express poorly,
  revisit AGE for that family specifically.
