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

![Two grids of as-of dates by customers: the dense run computes every cell, 18 of 18; the paired run computes the 7 declared pairs](/featurizer/images/paired-cohort-grid.svg)

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

The matrix then has one row per declared pair, for a target with one row per
id. A pair that appears twice still gives one row. For a target with several
rows per id (an event-like target, one row per visit with the patient as its
`id`), a pair keeps every row of that id the date can see.

`featurizer validate` checks the block's shape: a mapping with the one key
`id_column`, a non-empty string, on a target that declares an `id`. It cannot
check that your table has the column, because validation has no database
connection. A wrong name surfaces when the query runs, as PostgreSQL's
`column _cohort.<name> does not exist`.

## Two recipes for the pair table

The pair table is yours to write, and its shape is where the cohort is
defined. Two recipes cover the cases that come up. Both are SQL you run before
featurizer, on the same connection when the table is `TEMP`; both were run
against the test database with the rows below.

### Entities active before each date

Score, at each month end, the customers who had an order in the month before
it:

```sql
create temp table as_of_dates as
select d::date as as_of_date, active.customer_id as cohort_id
from generate_series(date '2024-02-01', date '2024-04-01', interval '1 month') as d
join lateral (
  select distinct customer_id
  from orders
  where ordered_at <  d::date
    and ordered_at >= d::date - interval '1 month'
) active on true;
```

With orders on 2024-01-10 (customer 1), 2024-02-20 (customer 2) and
2024-03-05 (customer 1), the table holds `(2024-02-01, 1)`, `(2024-03-01, 2)`,
`(2024-04-01, 1)`, and the matrix has exactly those three rows. This is the
shape `benchmarks/final_matrix.py --dates N` builds on the live databases.

### The events of a date

A model scores each event once, on its own date, with the history that
preceded it: a game scored with its teams' previous games, a request scored with its
area's earlier requests, a visit scored with the patient's earlier visits. The
target is the event table, and the pair table is one row per event:

```sql
create temp table as_of_dates as
select played_on - 1 as as_of_date, game_id as cohort_id
from games;
```

![Timeline of one game and its home team's earlier games: paired with the day before, the game's row is emitted and only the team's rows before that day are read; the team's row on the game's own day and a later one are not](/featurizer/images/events-of-a-date.svg)

Two decisions sit in that statement.

**The as-of date is the day before the event.** With `as_of_boundary:
inclusive` (the default) a child row dated on the as-of date is knowable, so
pairing a game with its own date lets the teams' rows of that same day into
the game's features. Paired with the day before, a game sees everything up to
and including the previous day, and its own day counts for the next game.

**The target declares no `temporal_ix`.** A target with one is read as of the
date ([ADR-0017](/featurizer/engineering/adr/0017-an-unknowable-row-is-not-emitted/)),
and an event dated after its as-of date is not emitted under it, which is the
day before by construction. Without one the target is read whole, and the
pairing alone decides which events a date returns. The children keep their
`temporal_ix`, and that is what the cut applies to.

With games on 2024-03-01 (home team 100), 2024-03-01 (team 200), 2024-03-08
(team 100) and 2024-03-15 (team 300), and the teams' rows on 2024-02-20,
2024-03-01, 2024-03-01, 2024-03-08 and 2024-03-15, the matrix has one row per
game: the first game's `COUNT(home.played_on)` is 1 (the 2024-02-20 row), the
third's is 2, and the second and fourth, whose home teams had no earlier row,
are NULL.

If the event's own day should count, pair the event with its own date and
declare the `temporal_ix` on the target: the event's row is then emitted with
every child row up to and including that day (and so is every earlier row of
the same id, for a target with several rows per id), while its rows dated
later are cut. Which of the two is right is a modelling decision about what
was known when the prediction was needed, and featurizer renders either.

## What it renders

Three things change, and only when the block is present.

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

A child that only the target aggregates is read for the cohort's rows too, so a
date no longer aggregates the history of every entity to keep a twentieth of
it:

```sql
orders_synth as (
  select …
  from orders
  where orders."order_id" in (
          select _rows."order_id" from orders _rows
          where _rows."customer_id" in (
            select _cohort."cohort_id" from as_of_dates _cohort
            where _cohort.as_of_date = aod.as_of_date))
    and orders."ordered_at" <= aod.as_of_date
)
```

The shape of that predicate follows what the child's rows are used for:

- **Whole window partitions.** A window transformer partitions by the child's
  `id`. Nothing says an id stays under one customer, so the rows kept are every
  row whose id has a row in the cohort, as above. When the child's `id` *is* the
  join key, or the child declares no `id` (then it has no window), the cut is
  the plain `orders."customer_id" in (…)`.
- **A join on another column of the target** (a request's `community_area`)
  goes through the target: the keys kept are that column's values over the
  cohort's rows.
- **Two relationships to the target** (a game's home team and its away team)
  keep the rows of either.

## Which children are narrowed

A child is narrowed only when the target's aggregations are the *only* readers
of its rows. featurizer decides that from the plan it has just built, not from
the config, because depth and traversal order decide which relationships are
consumed. So these keep their full read:

- a child that a second parent also aggregates (a region's order count needs
  the orders of customers the cohort does not name);
- an entity that another entity looks up, with or without `temporal: as_of`;
- a grandchild: its reader is the child, not the target;
- every entity, when a population-level transformer is selected (below).

Peer groups, spatial relationships and graph relationships read base tables in
CTEs of their own. Narrowing a synth does not reach them.

## What it guarantees

- **The values equal the dense run's on the declared pairs.** The
  integration tests run both and compare them: through the single-query path,
  through the temp-table path, with peer groups, with a population-level
  transformer, and once for every registered transformer on the target. For the
  narrowed children they compare twelve graph shapes (each predicate form, a
  second parent, a looked-up entity, a grandchild, an as-of lookup on the child,
  a dated target), every registered transformer on a narrowed child in three id
  shapes, and every registered aggregation over one. A cut on the join key
  alone fails 37 of the 81 cases in which an id crosses entities; the
  whole-partition cut fails none.
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

Measured 2026-09-21 on master plus this change, PostgreSQL 16.14, `jit` off (the
engine turns it off itself since #53):

| database | config | dates | dense rows | paired rows | dense | paired |
|---|---|---|---|---|---|---|
| dirtyduck | 147 features | 6 | 133,014 | 7,070 | 5.1 s | 0.4 s |
| dirtyduck | 272 features, 65 aggregations | 2 | 44,338 | 2,184 | 6.8 s | 1.7 s |
| dirtyduck | 272 features, 65 aggregations | 6 | 133,014 | 7,070 | 21.3 s | 5.0 s |
| dirtyduck | 1,252 features | 2 | 44,338 | 2,184 | 12.7 s | 2.0 s |
| donorschoose | 175 features | 6 | 18,000 | 1,086 | 0.9 s | 0.1 s |
| donorschoose | 1,063 features | 2 | 6,000 | 308 | 14.0 s | 3.5 s |
| chicago311 | 28 features | 6 | 183,924 | 183,924 | 1.1 s | 1.3 s |
| chicago311 | 191 features | 2 | 61,308 | 61,308 | 5.1 s | 5.0 s |

chicago311 is the worst case on purpose. Its children are keyed by community
area and request type, every area has an event every month, so the cohort the
harness derives is the whole population and the cut removes nothing. There is
no gain, and 0.2 s of overhead on the smallest config.

Before the child reads were narrowed, the second row was 7.8 to 8.4 s dense and
6.4 to 6.7 s paired: the pairing saved the target's side only, and each date
still aggregated the child rows of all 22,169 entities to keep about 1,100 of
them.
An earlier version of this page reported 48.6 s and 52.4 s for that cell. Most
of each was PostgreSQL's JIT compiling the target list: the same commit, re-run
on 2026-09-21, takes 31 s with `jit` on and 7.7 s with it off (#53).

**On the values.** The narrow configs agree with the dense run on every pair.
The wider ones agree on every row whose child timestamps are distinct. On
dirtyduck 9 of 2,184 rows differ, all of them entities with two inspections on
one date: an order-dependent aggregation over tied timestamps returns a value
that depends on the physical order of the rows, in the dense query as well
(issue #66), and a narrowed read changes that order. donorschoose adds 2 rows
where `cosinor_amplitude_weekly` divides by rounding noise (issue #67). The
harness records both counts in its artifact.

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

The values still equal the dense run's. featurizer narrows nothing upstream,
no child either, so the only saving is in the rows returned. A custom transformer that windows across
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
