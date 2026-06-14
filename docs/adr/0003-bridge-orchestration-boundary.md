# 0003 — The φ-bridge is an orchestration-agnostic library

**Status:** Accepted

**Date:** 2026-06-14

**Deciders:** Adolfo De Unánue

## Context

The φ-bridge ([[0001-phi-bridge-precompute-causal-boundary]]) does real work:
it reads source rows, runs heavy Python, and writes a table. The pipeline
standard for these projects is Dagster/Snakemake + dbt — so the question was
whether the bridge should embed scheduling, retries, and asset wiring.

## Decision

`featurizer/bridge/` is a plain library: `BridgeComputer.materialize(conn, ...)`
takes a live connection and does its I/O synchronously when called. It schedules
nothing, owns no assets, and imports no orchestrator. Wiring it as a Dagster
asset or Snakemake rule upstream of the SQL run is the caller's responsibility.

## Consequences

- The bridge stays testable with a bare psycopg connection and reusable from any
  orchestrator (or none) without inversion-of-control ceremony.
- It does **not** satisfy the pipeline-orchestration standard on its own; a
  project using it in production must place the `materialize` call inside a
  Dagster asset / Snakemake rule, not call it from an ad-hoc script.
- Optional heavy dependencies live in the `[bridge]` extra, so importing the SQL
  spine never drags in spaCy / sentence-transformers / networkx.
