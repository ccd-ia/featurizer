# 0004 — Peer groups are defined by a categorical column

**Status:** Accepted

**Date:** 2026-06-14

**Deciders:** Adolfo De Unánue

## Context

Peer-group features compare each entity to "others like it" as-of a date. The
peer set could be defined three ways: (A) a shared categorical column, (B) an
explicit peer-pair table, (C) computed similarity (k-NN). The full analysis is in
[`docs/peer-group-model-alternatives.org`](../peer-group-model-alternatives.org).

## Decision

Start with **Option A**: `peer_groups: [{by: <categorical column>, measures: […]}]`.
Peers are rows of the same entity sharing the `by` value; features are
leave-one-out and bounded `<= aod.as_of_date` on both membership and the peers'
child stream. The config is shaped so B is a clean superset later (it would reuse
the edge-table machinery) and both feed the same feature emitter; C is deferred
behind the φ-bridge.

## Consequences

- Lowest leakage surface (membership is a static column; only the event stream
  is time-bounded) and zero extra data — the seeded datasets already support it.
- Fully verifiable: the peer aggregate is a `GROUP BY` an independent SQL query
  can reproduce, which the realistic tier asserts.
- Peers must be expressible as one shared column; "similar but not identical"
  peers wait for Option B (curated table) or C (similarity graph, needs M2).
