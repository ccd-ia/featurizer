---
title: FAQ & troubleshooting
description: >-
  Answers to the recurring "why" and "what went wrong" questions about
  featurizer — installation, PostgreSQL-only design, point-in-time correctness,
  the 1664-column limit, categorical vocabularies, and the docs build.
sidebar:
  order: 2
---

The questions below are the ones the rest of the documentation already answers,
collected in one place. Each answer links to its canonical source — an ADR, a
reference page, or the code — so you can go deeper. If your question isn't here,
open an [issue](https://github.com/ccd-ia/featurizer/issues).

## Installation & compatibility

### How do I install it? Is it on PyPI?

There is **no PyPI package — deliberately.** The name is generic (featurizer
derives from [`dssg/featurizer`](https://github.com/dssg/featurizer)), so
distribution goes through **GitHub releases on `ccd-ia/featurizer`** instead.
Three ways to install:

```bash
# 1. From source (the development path)
git clone https://github.com/ccd-ia/featurizer && cd featurizer && uv sync

# 2. Straight from git into your own project
uv add "git+https://github.com/ccd-ia/featurizer.git"
#   or: pip install "git+https://github.com/ccd-ia/featurizer.git"

# 3. A pinned wheel from a tagged release
#   grab the .whl asset from github.com/ccd-ia/featurizer/releases
```

Optional extras live behind `uv sync --extra <name>`: `viz` (diagnostic plots),
`bridge` (the φ-bridge precompute companion), `parquet` (Arrow output). The SQL
spine never imports them.

### Does it work with MySQL, SQLite, DuckDB, or BigQuery?

**No — featurizer emits PostgreSQL-dialect SQL and is validated only against
PostgreSQL.** The generated queries lean on Postgres-specific features:
`LEFT JOIN LATERAL` for as-of joins, ordered-set aggregates (`percentile_cont`,
`mode() within group`), `bool_and`/`bool_or`, PostGIS `ST_*` for spatial
features, and `enum` catalog introspection for categoricals. Run the SQL on a
real PostgreSQL instance (the test suite spins up an ephemeral `postgres:16`
container via `just db-up`).

## Concepts

### What does "as-of" / point-in-time correctness mean, and why should I care?

A feature is a function `φ(entity, t)` that may only see events with timestamp
`τ ≤ t`. If a feature computed for a training row at date `t` accidentally reads
events from *after* `t`, that's **data leakage**: the model looks brilliant in
backtest and fails in production. Featurizer's temporal joins
(`mode: as_of`, with an optional `grace` lookback) enforce the `τ ≤ t` boundary
in SQL so leakage can't creep in. See the
[φ theory page](/featurizer/concepts/phi-theory/) and drag the as-of date in the
[interactive explorable](/featurizer/explorables/phi-dfs.html).

### Why aren't peer-group, spatial, or φ-bridge features in the primitives list?

Because they aren't registry primitives — they're **planner passes** driven by
their own config blocks (`peer_groups`, `spatial_relationships`, the native
1-hop `graph_relationships` pass added in 0.9.0) or **φ-bridge families** (the
`featurizer/bridge/` companion: sentiment, NER counts, readability, language
id, multi-metric centralities, Louvain community, embeddings, Markov
surprisal). Aggregations and transformers apply uniformly across the entity
graph; these families need cross-entity, second-table, or heavy-Python context
the registry model doesn't express, so they're deliberately separate. The
[primitives reference](/featurizer/reference/primitives/) covers everything
that *is* a registry primitive; the
[primitives explorer](/featurizer/explorables/primitives.html) lets you filter
and search them; the
[bridge cookbook](/featurizer/engineering/bridge-cookbook/) shows how to wire
and extend the bridge families.

## Common errors & limits

### `target list can have at most 1664 entries` — my wide config fails

PostgreSQL caps a CTE/result target list at **1664 columns**. A wide config —
many primitives × many intervals × many variables — blows past that in a single
monolithic query. The fix is **column-group sharding**: featurizer splits the
feature set into groups, materializes each group's CTE closure separately, and
re-joins on the full key. It kicks in automatically past a threshold you can
tune with `Featurizer(..., materialize_threshold=N)`. See
[performance internals](/featurizer/engineering/internals/) and
[ADR-0005](/featurizer/engineering/adr/0005-column-group-sharding/).

### My column names look truncated or contain a `~`

Generated feature names can exceed PostgreSQL's **63-byte identifier limit**.
Featurizer hash-truncates anything longer to a stable, collision-safe name
(quoted, so it's always a valid identifier). Internal CTE names use `_` as the
cap separator — a bare `~` there was a real bug (fixed in v0.8.0's companion-CTE
path). Your *output* column names follow the
[fixed one-hot naming contract](/featurizer/engineering/adr/0007-direct-categorical-fixed-vocabulary/)
and stay readable.

### A categorical one-hot column is missing, or has a value my data never contains

That's by design. Featurizer builds the categorical vocabulary from the
column's **PostgreSQL `enum` labels — it never scans the data** to discover
values. This makes the feature matrix **split-blind**: the same columns appear
whether you featurize the train split, the test split, or a single row, so
train/serve schemas can't drift. A value present in your enum but absent from a
given slice still gets its (all-zero) column; a value in your data but not the
enum is a modeling error to fix upstream. See
[ADR-0007](/featurizer/engineering/adr/0007-direct-categorical-fixed-vocabulary/)
and the [categoricals notebook](/featurizer/notebooks/05-categoricals-output/).
Imputation of the resulting matrix is **opt-in**, not automatic.

## Development & docs

### Why do the tutorial notebooks show outputs but never execute in CI?

The docs site renders each notebook from its **committed, executed outputs** —
the outputs that were validated against a live database — and never runs it
during the build. The GitHub Pages workflow has no PostgreSQL, and re-executing
would either fail or silently diverge. The committed outputs are the source of
truth; a [count-parity test](https://github.com/ccd-ia/featurizer/blob/master/tests/test_site_gen.py)
guards the generated pages against drift.

### How do I run the tests? Some are skipped

Tests are tiered via `just`:

```bash
just test-fast           # DB-free — runs anywhere
just db-up               # ephemeral postgres:16 container
just test-integration    # needs the database
just seed && just test-realistic   # realistic-dataset tier (integration + slow)
```

Integration tests **skip automatically** when no database is configured — that's
expected on a fresh checkout, not a failure.

### Why does basedpyright ignore `aggregations.py` and `transformations.py`?

Those two modules define the primitive variants with heavy dynamic patterns
(metaprogrammed classes, SQL-string templating) that the type checker can't
follow without noise. They're listed in `pyrightconfig.json`'s ignore set on
purpose; the rest of the codebase type-checks under `standard` mode.
