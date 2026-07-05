# 0008 — Relationship naming: only-when-ambiguous, explicit, and loud

**Status:** Accepted

**Date:** 2026-07-05

**Deciders:** Adolfo De Unánue

## Context

Feature names (`SUM(orders.amount|interval=P1M)`) and aggregation CTE names
(`orders_aggs_for_customers`) were derived from the **entity pair** only. Two
relationships between the same pair — customers→orders via `buyer_id` and via
`seller_id` — therefore collided; the traversal guard compounded this by
consuming only the first relationship that reached an already-built entity, so
the second leg (and the second path of a diamond `a←b←d` / `a←c←d`) silently
vanished from the output matrix.

Feature names are a **downstream contract** (ADR-0007): they become Parquet
columns, persisted feature-group table columns, and feature-importance keys in
triage-pg. Any fix that renames existing features breaks consumers.

## Decision

Relationships accept an optional `name:` (config `relationships[].name`), used
as the *naming alias* in aggregation feature names (`SUM(purchases.amount)`),
in CTE names (`purchases_aggs_for_customers`), and — for named forward
transfers — as a column qualifier (`"purchases.score"`).

- **Unique (parent, child) pair** (every config valid before v0.5.0): `name`
  is optional and the alias defaults to the entity alias — every existing
  feature name stays **byte-identical**.
- **Repeated pair**: validation **errors** unless every such relationship
  carries a distinct `name:`. Loud, not silent.
- Entities build once; **every relationship is consumed** (the planner
  snapshots each entity's built feature set so later legs aggregate exactly
  what the entity's transform projects). Only true cycles (an in-progress
  ancestor) skip consumption.

## Alternatives rejected

- **Always-qualify** (`orders[buyer_id]` everywhere): renames every existing
  feature — breaks the ADR-0007 downstream contract for zero benefit in the
  99% unambiguous case.
- **Auto-suffix only on collision**: still silently renames leg 1 the day a
  second relationship is added; explicit names make the churn visible in the
  config diff and reviewable.
- **Warning + auto-dedupe**: half-loud is still a silently wrong matrix for
  anyone not reading logs.

## Consequences

- Parallel relationships and diamond topologies now produce complete, distinct
  feature sets; adding a second relationship to an existing pair is a
  validation-guided (named) change, never a silent rename.
- `name` must be a valid identifier, unique among relationship names, and
  distinct from entity aliases (it becomes a CTE-name segment).
- The planner's `_built_features` snapshot is the source of truth for what a
  relationship may aggregate; the live `_features` set accumulates
  consumer-bound features and must not be re-consumed.
