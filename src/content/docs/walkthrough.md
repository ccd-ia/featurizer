---
title: "Walkthrough: zero to a feature matrix"
description: >-
  Install featurizer, describe your tables in one YAML file, and materialize a
  point-in-time-correct feature matrix from PostgreSQL — in about 15 minutes.
sidebar:
  order: 1
---

featurizer implements [Deep Feature Synthesis](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf)
(Kanter & Veeramachaneni, IEEE DSAA 2015) for temporal, relational data —
compiled to **pure PostgreSQL**. You describe entities and relationships in
YAML; featurizer plans, renders, and (optionally) executes SQL that computes
hundreds of point-in-time-correct features. The
[φ formalism](/featurizer/concepts/phi-theory/) explains *why* this
construction cannot leak.

This walkthrough follows [example 01](https://github.com/ccd-ia/featurizer/tree/master/examples/01-basic-aggregations)
(an e-commerce scenario: customers and their orders). Clone the repo to run
every step verbatim:

```bash
git clone https://github.com/ccd-ia/featurizer.git
cd featurizer
```

## 1. Install

featurizer is distributed through
[GitHub releases](https://github.com/ccd-ia/featurizer/releases) — deliberately
no PyPI. Pin a tag:

```bash
# with uv (recommended)
uv add "featurizer @ git+https://github.com/ccd-ia/featurizer.git@v0.8.0"

# or with pip
pip install "featurizer @ git+https://github.com/ccd-ia/featurizer.git@v0.8.0"
```

Add the `[parquet]` extra (`featurizer[parquet] @ …`) if you want Arrow/Parquet
output. Working inside the cloned repo, `uv sync` is all you need.

## 2. The data

Three tables. Two are yours; one small one is featurizer's contract:

| table | role |
|---|---|
| `customers` | the **target** entity — one row per customer (`customer_id`, `signup_date`, `country`, `age`) |
| `orders` | a **child** event stream — one row per order (`order_id`, `customer_id`, `order_date`, `amount`, `status`) |
| `as_of_dates` | one `as_of_date` column: the snapshot dates you want features *as of* |

![Entity-relationship diagram: customers 1-to-many orders, both filtered by the as_of_dates spine](/featurizer/images/walkthrough-erd.svg)

The column featurizer cares most about is each entity's **`temporal_ix`** — the
event timestamp (`order_date` for orders). Every generated feature is computed
using only rows with `temporal_ix <= as_of_date`, which is what makes the
matrix **point-in-time correct**: a feature for June 1st cannot see June 2nd,
so nothing leaks from the future into training data.

## 3. One YAML file

The whole configuration for this scenario
([`examples/01-basic-aggregations/config.yaml`](https://github.com/ccd-ia/featurizer/blob/master/examples/01-basic-aggregations/config.yaml)):

```yaml
target: customers          # the entity features are FOR
max_depth: 2               # how far to traverse relationships

intervals:                 # rolling windows, ISO-8601 durations
  - P7D                    # last 7 days
  - P30D                   # last 30 days

aggregations:              # a focused set keeps the tutorial readable —
  [count, sum, mean, min, max, stddev, nunique]
transformations:
  [identity, abs]

entities:
  - alias: customers
    id: customer_id
    table: customers
    temporal_ix: signup_date
    variables:
      country:
        type: categorical
        role: categorical              # one-hot against a FIXED vocabulary —
        vocabulary: [AU, CA, DE, FR, UK, US]   # split-blind, fit-free
      age:
        type: numeric

  - alias: orders
    id: order_id
    table: orders
    temporal_ix: order_date
    variables:
      amount: {type: numeric}
      status: {type: categorical}

relationships:
  - parent: {entity: customers, key: customer_id}
    child:  {entity: orders,    key: customer_id}
```

Omit `aggregations:`/`transformations:` and featurizer applies its full default
set — 67 aggregations × 83 transformers, which is usually far more than a
tutorial (or PostgreSQL's 1664-column row limit) wants.

## 4. Render the SQL — no database needed

```bash
uv run python examples/01-basic-aggregations/run_example.py --show-sql
```

or in Python:

```python
from featurizer import Featurizer

f = Featurizer("examples/01-basic-aggregations/config.yaml")
print(f.query)          # a single PostgreSQL query — inspect before you run
```

For this config that is one 56-line, ~19&nbsp;KB query. Its skeleton is worth
reading once, because every featurizer query has this shape:

```sql
select aod.as_of_date, t.*
from as_of_dates as aod
cross join lateral (
  with
    orders_synth as (…),          -- child columns, selected
    orders_transform as (…),      -- transformers applied (abs(amount), …)
    orders_aggs_for_customers as (
      select customer_id,
        count(order_id)             as "COUNT(orders.order_id)",
        count(order_id) filter (where daterange((aod.as_of_date
              - interval 'P7D')::date, aod.as_of_date::date, '[]')
              @> order_date::date)  as "COUNT(orders.order_id|interval=P7D)",
        sum(amount)                 as "SUM(orders.amount)"
        -- … one column per (aggregation × variable × interval)
      from orders_transform
      where order_date <= aod.as_of_date     -- the leakage guard
      group by customer_id
    ),
    customers_synth as (…),       -- join aggregates onto the target
    customers_transform as (…)    -- target-level transformers + one-hots
  select * from customers_transform
) as t
```

The `cross join lateral` re-evaluates the feature CTEs **per as-of date**, and
the `where order_date <= aod.as_of_date` guard plus interval `filter` clauses
are the point-in-time semantics, visible in plain SQL.

## 5. Execute against PostgreSQL

featurizer emits PostgreSQL-dialect SQL, so execution needs a real PostgreSQL.
The repo manages a throwaway one:

```bash
just db-up                                              # ephemeral postgres:16
uv run python examples/01-basic-aggregations/create_data.py   # seed example_01
uv run python examples/01-basic-aggregations/run_example.py --execute
```

(Any PostgreSQL works: set `DATABASE_URL` or the `PG*` variables instead.)
In Python, materialization is one call:

```python
df = f.to_dataframe()
df.shape        # (1200, 105) — 100 customers × 12 as-of dates, 105 features
```

The frame is indexed by `(as_of_date, customer_id)`: the same customer appears
once per snapshot date, with features computed from only what was knowable at
that date.

## 6. Read your features

Column names are self-describing — `AGG(entity.column|interval=WINDOW)`:

```text
COUNT(orders.order_id|interval=P7D)     orders in the last 7 days
SUM(orders.amount|interval=P30D)        spend in the last 30 days
MEAN(orders.ABS(orders.amount))         mean of a transformed child column
customers.country=US                    fixed-vocabulary one-hot (0/1)
```

Two conventions to know:

- **NULL is signal.** A customer with no orders in the window gets `NULL`
  (not 0) for `MEAN(orders.amount|interval=P7D)` — "no data" and "zero" are
  different facts. Opt into imputation explicitly via
  `to_dataframe(impute=True)` when you want it.
- **Every column is documented by the feature manifest.**
  `f.feature_manifest` (also persisted as a `<target>_manifest` table by
  `to_tables()`) carries, per column: the full label, kind
  (variable / one_hot / derived), lineage (depth, parents, source column,
  interval) and a generated plain-language description.

## 7. Choose your primitives

List everything the registry offers — 67 aggregations and 83 transformers:

```bash
uv run python -m featurizer list-primitives --type agg --category
uv run python -m featurizer list-primitives --type transform --show-sql
```

Then select per config, as example 01 does. Beyond the basics there are
ordered-set aggregations (`median`, `p90`…), temporal-gap statistics
(`gap_mean`, `burstiness`), categorical distributions (`entropy`, `hhi`,
`gini`), and sequence features (`ngram_2_freq`, `longest_streak`) — the
[primitives reference](/featurizer/reference/primitives/) lists all 150 with
SQL examples.

## 8. Point-in-time joins (as-of)

When a *parent* record should contribute the **most recent state as of each
snapshot** — a patient's latest care plan, a school's history — declare the
relationship temporal:

```yaml
relationships:
  - parent: {entity: patients,   key: patient_id}
    child:  {entity: care_plans, key: patient_id}
    temporal:
      mode: as_of
      grace: P21D        # optional: only look back this far
```

featurizer renders a `left join lateral … order by … limit 1` that picks the
newest child row at or before each `as_of_date`. Tutorial 02 (healthcare)
works through this in depth —
[examples/02-temporal-joins](https://github.com/ccd-ia/featurizer/tree/master/examples/02-temporal-joins).

## 9. Visualize the matrix

The optional `[viz]` extra adds `FeaturizerViz` — distribution, missingness,
correlation, embedding, and per-entity temporal diagnostics on the materialized
matrix:

```python
from featurizer import FeaturizerViz

viz = FeaturizerViz.from_featurizer(f, df=df)
viz.plot_feature_distributions(kind="violin")
viz.plot_missing_heatmap()          # NULL-as-signal, made visible
```

Real output from a live 177k-row × 272-feature matrix:

![Violin plots of top-variance features across entities](/featurizer/images/viz/feature-distributions.png)

![Missingness heatmap — NULLs kept as signal](/featurizer/images/viz/missing-heatmap.png)

## 10. Where next

- **The tutorials**: five executed notebooks, from basic aggregations to custom
  primitives —
  [examples/](https://github.com/ccd-ia/featurizer/tree/master/examples)
  (rendered versions join this site shortly).
- **The theory**: [φ — the formalism behind feature creation](/featurizer/concepts/phi-theory/),
  with an interactive explorable.
- **References**: the full
  [primitive registry](/featurizer/reference/primitives/) and the complete
  [`config.yaml` schema](/featurizer/reference/configuration/).
- **Proof it holds up**: every release is validated against three live
  databases — <a href="/featurizer/specs/live-db-revalidation-v080.html">the
  v0.8.0 reports</a>.
