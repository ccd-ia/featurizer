---
title: "Paired cohorts"
description: >-
  When each as-of date has its own set of entities, declare the pairing so the
  matrix holds only those rows. What the default cohort is, when it is the
  wrong shape, what the pairing renders, and what it does and does not save.
sidebar:
  order: 2
  label: Paired cohorts
---

featurizer computes every feature *as of* a date. The dates come from a table
you provide, `as_of_dates`, and the default reading of that table is dense:
featurizer emits every row of the target entity under every date.

```text
as_of_dates          target            matrix
-----------          ------            ---------------------------
2024-02-01      x    entity 1     =    (2024-02-01, 1) (2024-02-01, 2) (2024-02-01, 3)
2024-03-01           entity 2          (2024-03-01, 1) (2024-03-01, 2) (2024-03-01, 3)
                     entity 3
```

## The default is right for the standard shape

Temporal cross-validation scores the same population at a series of dates: every
customer at each month end, every facility at each quarter. There the dense
product *is* the wanted matrix, and nothing on this page applies.

## When each date has its own entities

Some cohorts are defined by the date. An event that happens on a date has the
entities that took part in it. A daily scoring run covers that day's arrivals.
The entity set changes from one date to the next, and an entity may appear under
one date only.

Under the dense reading those cohorts cost `dates x entities` rows to compute
and keep one row per pair. With 6 dates and 22,169 entities the dense matrix has
133,014 rows; if the six cohorts hold 7,070 pairs between them, PostgreSQL
computes 19 of every 20 rows for the caller to throw away. The ratio grows with the number of
dates.

## The pairing declaration

Give `as_of_dates` a second column that holds the target's id, one row per
pair, and name that column in the config:

```sql
create table as_of_dates (as_of_date date, cohort_id bigint);

insert into as_of_dates values
  ('2024-02-01', 1), ('2024-02-01', 2),
  ('2024-03-01', 2), ('2024-03-01', 3);
```

```yaml
target: customers
as_of_dates:
  id_column: cohort_id     # the column of as_of_dates that holds customers' id
```

The matrix then has exactly one row per declared pair. A pair that appears twice
still gives one row.

`featurizer validate` checks the block's shape: a mapping with the one key
`id_column`, a non-empty string, on a target that declares an `id`. It cannot
check that your table has the column, because validation has no database
connection. A wrong name surfaces when the query runs, as PostgreSQL's
`column _cohort.<name> does not exist`.

## What it renders

Two things change, and only when the block is present.

`aod`, the alias featurizer computes every feature against, ranges over the
*distinct* dates of the table, because a pair table repeats each date:

```sql
from (select distinct as_of_date from as_of_dates) as aod
cross join lateral ( … ) as t
```

The target's base read keeps the ids paired with the date `aod` is on:

```sql
customers_synth as (
  select …
  from customers
  left join orders_aggs_for_customers on …
  where customers."customer_id" in (
    select _cohort."cohort_id" from as_of_dates _cohort
    where _cohort.as_of_date = aod.as_of_date)
)
```

The cut is in the read, not on the result, so the target's transform and
everything after it run on the cohort's rows only.

## What it guarantees

- **The values equal the dense run's on the declared pairs.** The
  integration tests run both and compare them: through the single-query path,
  through the temp-table path, with peer groups, with a population-level
  transformer, and once for every registered transformer.
- **Without the block nothing changes.** A config that does not declare
  `as_of_dates` renders SQL byte-identical to the SQL it rendered before the key
  existed. The test suite compares SHA-256 digests of the single query, the
  column-group queries and the temp-table statements against digests captured
  from the commit before the change.

## What it saves, measured

`benchmarks/final_matrix.py --dates N` runs the dense and the paired query on
one of the project's live validation databases (22,169 entities, monthly as-of
dates, each date paired with the entities that had an event in the month before
it) and checks the two agree on the pairs.

| config | dates | dense rows | paired rows | dense | paired |
|---|---|---|---|---|---|
| 147 features | 6 | 133,014 | 7,070 | 8.3 s | 3.0 s |
| 272 features, 65 aggregations | 2 | 44,338 | 2,184 | 48.6 s | 52.4 s |

Both runs returned the same values on the pairs, with no column differing.

The saving comes from the target's side of the query: its base read, its
transform, and the rows sent back. **The child aggregations are not narrowed.**
Each date still aggregates the child rows of every entity, as the dense query
does. A config whose cost is mostly aggregation therefore gains nothing in
query time: the second row is 8% slower than dense, and an earlier run of the
same pair of queries was 4% slower. What that config does gain is a matrix 20
times smaller to fetch, store and join. Pushing the cohort into the child reads
is a separate piece of work: it is only valid for features that depend on an
entity's own rows, and that has to be established primitive by primitive.

## Population-level transformers

`cross_entity_zscore` and `cross_entity_percentile` compare a target row with
the other target rows (`avg(x) over ()`). Narrowing the base read would shrink
that population to the cohort and change the value. When a config selects one of
them, featurizer applies the same cut to the final select instead:

```sql
select * from customers_transform
where "customer_id" in (select _cohort."cohort_id" from as_of_dates _cohort
                        where _cohort.as_of_date = aod.as_of_date)
```

The values still equal the dense run's. featurizer narrows nothing upstream, so
the only saving is in the rows returned. A custom transformer that windows across
entities gets the same treatment by setting `population_level = True`.

## Why not one evaluation per pair

Letting `aod` range over the pairs, with `target.id = aod.cohort_id`, looks
simpler, and we measured it first. On the 147-feature config it was the fastest
shape, 0.8 s, because PostgreSQL pushes the equality down into every inlined
aggregation. On a 65-aggregation config PostgreSQL cannot inline the CTEs that
several aggregations share, so each of 2,184 pairs recomputed them in full: the
query had not finished after 1,520 s, where the dense query took 72 s. The
per-date shape evaluates the lateral once per date, exactly as the dense query
does, so its cost stays within a few percent of dense on every config measured.
That bound is why it is the one that ships.

See the [`as_of_dates` entry](/featurizer/reference/configuration/#top-level-keys)
in the configuration reference.
