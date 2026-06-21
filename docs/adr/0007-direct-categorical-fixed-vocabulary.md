# 0007 — Direct categoricals one-hot only from a fixed (declared/ENUM) vocabulary

**Status:** Accepted

**Date:** 2026-06-21

**Deciders:** Adolfo De Unánue

## Context

Direct categorical attributes on the target entity (e.g. a facility's
`facility_type`) previously passed through as raw string columns. A downstream
consumer (triage-pg) feeds the matrix into an sklearn pipeline, where a raw
string column crashes the encoding step — so featurizer needs to one-hot encode
these itself.

One-hot encoding needs a **vocabulary** (the set of category values → columns).
Where that vocabulary comes from is a leakage boundary, not a convenience choice:

- A vocabulary **learned from the data** (scan `distinct` values) is fit on
  whatever rows happen to be present. Featurizer is **split-blind** — it has no
  concept of train/test, cohorts, or as-of splits — so a data-derived vocabulary
  would silently include test-period categories, leaking them into what the
  consumer treats as a train-only transform.
- A vocabulary that is **declared or fixed** (a config list, or the column's
  PostgreSQL `ENUM` labels) is fit-free and deterministic: it does not depend on
  which rows are present, so it carries no split.

Featurizer must stay general (no triage / split / cohort / label concepts).

## Decision

Featurizer one-hot encodes a `role: categorical` direct variable **only** from a
fixed vocabulary, resolved in priority order: (a) a declared `vocabulary: [...]`
in the config; (b) the column's PostgreSQL `ENUM` labels (introspected via
`pg_enum`); (c) otherwise it **fails loud**. It **never** scans the data for
distinct values — that fitted, split-sensitive transform belongs to the consumer
(train-only), not to featurizer.

Mechanics:

- Vocabularies are resolved **once at `Featurizer` construction** (so the planner
  stays DB-free and `query` / `--show-sql` work without a database for declared
  vocabularies). `ENUM` introspection needs a connection: the optional
  `Featurizer(..., connection=)` is used, else one is opened from
  `DATABASE_URL` / `PG*`; if neither resolves, the declare-vocabulary-or-`ENUM`
  error is raised.
- The vocabulary is **sorted** for a deterministic, reproducible one-hot column
  order. Each value becomes a numeric `case when <col>::text = '<value>' then 1
  else 0 end` column — the `::text` cast and `else 0` make a NULL or
  out-of-vocabulary value an **all-zero** row rather than a crash or an
  invalid-enum-input error.
- **Column-naming contract** (a stable interface the consumer relies on): each
  one-hot column is named `"<entity_alias>.<column>=<value>"`, a quoted
  PostgreSQL identifier capped at 63 bytes by the existing `pg_identifier`
  hash-truncation. No prefix collides with the key columns (`as_of_date`, the
  target id) or the imputation `*__missing` suffix; the consumer strips those and
  treats the rest as numeric features. The full untruncated name is recoverable
  from `Featurizer.feature_manifest` (`column` ↔ `label`).
- A `role: identifier` direct variable (a name, license number, exact address) is
  excluded from the output, loudly. A raw `text`/`categorical` direct variable
  left with no role warns before passing through unencoded.

Scope: this applies to the **target entity's own** direct variables. Child-event
categoricals are unchanged (already reduced to numeric via aggregation).

## Consequences

- Featurizer remains split-blind and fit-free: it can encode declared/ENUM
  categoricals safely, and refuses (loudly) to do the one thing that would leak —
  learning a vocabulary from data. The train-only learned-vocabulary case stays
  the consumer's responsibility, a clean division triage-pg can rely on.
- The `"<entity>.<col>=<value>"` naming is now a **public contract**: renaming it
  is a breaking change for any consumer that keys on those columns. The manifest
  (`feature_manifest` / `manifest_dataframe()`) is the supported way to recover
  the full intended name when the 63-byte cap truncates a long label, and to get
  human-readable labels for partner tables and plots.
- Declared vocabularies keep the SQL-render path fully DB-free; only the
  `ENUM`-introspection path touches the database, and only to read `pg_enum`
  (never the data).
- A `role: categorical` variable with neither a declared vocabulary nor an
  introspectable `ENUM` fails at construction with actionable guidance, rather
  than silently emitting a raw string column or a learned vocabulary.
- **Not covered (deferred):** one-hot encoding of categoricals on *parent*
  entities reached via forward relationships (only target-owned today); and an
  optional one-line count of out-of-vocabulary data values (skipped to keep the
  path DB-light and strictly fit-free).
