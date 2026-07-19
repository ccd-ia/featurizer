# Featurizer Examples

This directory contains practical examples demonstrating key features of the Featurizer library.

> The tutorial notebooks are also **rendered on the docs site** (from their
> committed, executed outputs):
> [ccd-ia.github.io/featurizer/notebooks](https://ccd-ia.github.io/featurizer/notebooks/).

## Overview

Each example includes:
- **README.md** - Detailed explanation of the scenario and concepts
- **config.yaml** - Featurizer configuration file
- **create_data.py** - Loads sample data into PostgreSQL (its own schema)
- **run_example.py** - Generates features, prints SQL, and (with `--execute`) runs the query

Featurizer emits **PostgreSQL-dialect** SQL (lateral joins, `percentile_cont …
within group`, `::date` casts), so executing an example needs a PostgreSQL
database. The easiest way is the throwaway Docker container the repo's `justfile`
manages; you can also point `DATABASE_URL` / `PG*` at any PostgreSQL you have.
Generating and inspecting the SQL (`--show-sql`) needs **no** database.

Each example loads into its own schema (`example_01` … `example_04`) and runs
with that schema on the `search_path`, so the configs use bare table names and
each example keeps its own `as_of_dates`.

## Examples

### [01-basic-aggregations](./01-basic-aggregations/)
**Difficulty:** Beginner — E-commerce (Customers → Orders)

Basic parent-child aggregations (count, sum, mean, min, max, stddev, nunique),
time windows (P7D, P30D), feature naming. **Start here.**

### [02-temporal-joins](./02-temporal-joins/)
**Difficulty:** Intermediate — Healthcare (Patients → Care Plans)

As-of join semantics, grace periods, point-in-time generation, LATERAL SQL.
Also exercises rolling stats (`rolling_mean_7`, `rolling_median_7`,
`rolling_iqr_7`) — the ordered-set rolling stats render as correlated subqueries
(PostgreSQL forbids `OVER` on `percentile_cont`).

### [03-deep-nesting](./03-deep-nesting/)
**Difficulty:** Intermediate — Retail Supply Chain (Stores → Orders → Order Items → Products → Suppliers)

Multi-level relationships (depth=3), feature propagation across chains, CTE
structure. Tables are created in FK-dependency order (PostgreSQL resolves
`REFERENCES` at creation time).

### [04-custom-primitives](./04-custom-primitives/)
**Difficulty:** Advanced — Financial Analytics (Accounts → Transactions)

Creating and registering custom aggregations (`range`, `p95`) and
transformations (`log1p`, `zscore`, `bin`) against the current primitive API,
then selecting them in `config.yaml`.

### [05-categoricals-output](./05-categoricals-output/)
**Difficulty:** Advanced — Food Inspections (Facilities → Inspections)

Direct-categorical one-hot encoding (`role: categorical` / `role: identifier`)
against a fixed vocabulary, the feature manifest, the output formats
(`to_dataframe` / `to_arrow`), and the imputation contract (`__missing`,
count-vs-measure). Unlike examples 1–4, this one **executes** against
PostgreSQL, so it shows the actual feature matrix.

### [06-graph-text-bridge](./06-graph-text-bridge/)
**Difficulty:** Advanced — Coordination Detection (Authors → Posts)

The φ-bridge two-stage pipeline (ADR-0001/ADR-0014): `SentimentBridge`
reduces each post to a valence scalar (Path 1), `NearDuplicateEdgeBridge`
turns copy-paste text into an `(src, dst, ts)` edge table (Path 2), and
`CentralityBridge.materialize_snapshots` rebuilds the graph per as-of window
into `(node, as_of_date)` snapshot rows the spine trends. All bridge outputs
are persisted tables (`persist=True`) and `run_example.py` asserts
`config.yaml` equals the bridges' `emit_yaml()` fragments. **Executes**
against PostgreSQL.

## Quick Start

```bash
# From the repository root. Start the throwaway PostgreSQL once:
just db-up

# Seed + run a single example end to end (NAME is a prefix):
just example 01            # or 02, 03, 04, 05, 06
# ...or run all of them:
just examples

# Tear the database down when done (the container is ephemeral):
just db-down
```

Without `just` (point at any PostgreSQL via the environment):

```bash
export DATABASE_URL=postgresql://user:pass@host:5432/dbname   # or set PG* vars
uv run python examples/01-basic-aggregations/create_data.py    # load the schema
uv run python examples/01-basic-aggregations/run_example.py --execute
```

Inspect the generated SQL without any database:

```bash
uv run python examples/01-basic-aggregations/run_example.py --show-sql
```

## Learning Path

1. **Example 1** — basic concepts
2. **Example 2** — temporal joins (+ rolling stats)
3. **Example 3** — deep nesting
4. **Example 4** — custom primitives
5. **Example 5** — direct categoricals, output formats & imputation (executes on PostgreSQL)
6. **Example 6** — φ-bridges: text → edges → centrality snapshots → spine (executes on PostgreSQL)

## Common Patterns

### Configuration Structure

```yaml
target: entity_alias       # Target entity for features
max_depth: 2               # Maximum relationship depth

intervals:                 # Time windows for aggregations
  - P7D
  - P30D

aggregations:              # Optional: a focused set (defaults to the full set,
  - count                  # which can exceed PostgreSQL's 1664 columns-per-row
  - mean                   # limit on wide configs)
transformations:
  - identity

entities:                  # Entity definitions
  - alias: entity_name
    id: primary_key
    table: table_name
    temporal_ix: timestamp_column  # Optional
    variables:
      column_name:
        type: numeric|categorical

relationships:             # Parent-child relationships
  - parent:
      entity: parent_alias
      key: parent_key
    child:
      entity: child_alias
      key: child_key
    temporal:              # Optional
      mode: as_of
      grace: P7D
```

### Primitive selection (why the configs are curated)

With no `aggregations:` / `transformations:` keys, Featurizer uses the full
default set (69 aggregations, 83 transformers). On even a two-entity config that
synthesizes **thousands** of feature columns — past PostgreSQL's hard limit of
1664 columns per row, and far past anything legible in a tutorial. Each example
therefore selects a focused set; widen it as you like (mind the 1664 ceiling).

### The `as_of_dates` table

Every `create_data.py` creates an `as_of_dates` table — the time points at which
features are computed — in the example's schema:

```sql
CREATE TABLE as_of_dates (as_of_date DATE PRIMARY KEY);
```

The generated query reads it as `from as_of_dates as aod`, resolved via the
schema on the `search_path`.

## Troubleshooting

### "No PostgreSQL configured"
`--execute` needs a database. Run `just db-up`, or export `DATABASE_URL` / `PG*`.
`--show-sql` works without one.

### "✗ Error executing query" / empty results
Make sure you loaded the data first: `python create_data.py` (or `just example NN`)
against the same database `run_example.py` points at.

### Driver note
Execution goes through `records`/SQLAlchemy with the **psycopg3** driver — the
examples build a `postgresql+psycopg://…` URL (the project doesn't depend on
psycopg2) and pin the schema with a libpq `options=-csearch_path=<schema>`.

## Next Steps

1. Read the main `README.md` for architecture details
2. Explore `featurizer/primitives/` for available aggregations and transformations
3. Try your own datasets with a custom `config.yaml`
