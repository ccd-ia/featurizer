---
title: Configuration reference
description: >-
  Every key of featurizer's YAML configuration — entities, variables and
  roles, relationships, temporal joins, intervals, primitive selection,
  peer groups, spatial features and 1-hop graph features.
sidebar:
  order: 2
---

One YAML file describes your schema and what to synthesize. The annotated
skeleton, then every key in detail:

```yaml
target: customers            # required — entity alias features are FOR
max_depth: 2                 # required — relationship traversal depth
intervals: [P7D, P30D]       # required — ISO-8601 rolling windows

aggregations: [count, mean]  # optional — omit for the curated default set
transformations: [identity]  # optional — omit for the curated default set
as_of_boundary: exclusive    # optional — inclusive (default) | exclusive

entities:                    # required
  - alias: customers
    table: customers         # fully-qualified name works too (schema.table)
    id: customer_id
    temporal_ix: signup_date
    variables:
      age: {type: numeric}

relationships:               # optional
  - parent: {entity: customers, key: customer_id}
    child:  {entity: orders,    key: customer_id}
```

`Featurizer("config.yaml")` validates on load — structural checks, value
checks (ISO-8601 durations, known variable types), semantic checks (target
exists, relationship endpoints exist, temporal-join requirements) and
best-practice warnings (`max_depth > 5`, more than 10 intervals). Unknown
primitive names get a "did you mean?" from the registry. Disable with
`Featurizer("config.yaml", validate=False)` for legacy configs.

## Top-level keys

| key | required | meaning |
|---|---|---|
| `target` | yes | the entity alias the output matrix is indexed on: one row per `(as_of_date, target id)` |
| `max_depth` | yes | how many relationship hops the planner traverses from the target |
| `intervals` | yes | ISO-8601 durations (`P7D`, `P1M`, `P1Y`…). Each one adds a windowed variant of every aggregation, on top of the lifetime variant that is always emitted — see [Column budget](#column-budget) |
| `aggregations` | no | subset of registered aggregations to apply — see the [primitives reference](/featurizer/reference/primitives/); omit (or `null`) for the curated default set. An explicit `[]` suppresses the aggregation layer: zero aggregation features (since v1.0.1) |
| `transformations` | no | subset of registered transformers; omit (or `null`) for the curated default set. An explicit `[]` suppresses the transform layer — features pass through unchanged, identical to `[identity]` (since v1.0.1) |
| `as_of_boundary` | no | `inclusive` (events at the as-of date count) or `exclusive` (strictly before) |
| `entities` | yes | the tables — see below |
| `relationships` | no | foreign-key links — see below |
| `spatial_relationships` | no | second-table spatial features — see below |
| `graph_relationships` | no | native 1-hop graph features over an edge table (0.9.0) — see below |

The runtime also expects an **`as_of_dates` table** (one `as_of_date` column)
in the database at execution time — it is the outer spine every feature is
computed *as of*.

## Entities

```yaml
entities:
  - alias: facilities          # short name used in CTEs and feature names
    table: clean.facilities    # the actual table (may be schema-qualified)
    id: facility_id            # primary key (optional, required for targets)
    temporal_ix: license_date  # event-timestamp column (optional)
    spatial_ix: location       # PostGIS point column (optional; spatial pass)
    variables:
      risk: {type: numeric}
      facility_type:
        type: categorical
        role: categorical      # one-hot against a FIXED vocabulary
        vocabulary: [Restaurant, Grocery Store, School]
      name:
        type: text
        role: identifier       # carried through, never a feature
```

- **`temporal_ix`** is what makes features point-in-time correct: interval
  aggregations and as-of joins filter on it. An entity without one
  contributes only static (non-windowed) features.
- **Variable `type`**: `numeric`, `categorical`, `text`, `boolean`, `date`,
  `timestamp`, or `index`. Types decide which aggregations/transformers
  apply.
- **`role: categorical` + `vocabulary`** one-hot encodes a *direct*
  categorical into `"<alias>.<col>=<value>"` 0/1 columns against the declared
  vocabulary (or the column's PostgreSQL `ENUM` labels if you omit
  `vocabulary`). Deliberately **split-blind and fit-free**: featurizer never
  learns a vocabulary from data — learned (train-only) encodings are the
  consumer's job.
- **`role: identifier`** excludes a column from the feature output while
  keeping it available as a key.

### Peer groups (planner pass)

Compare each entity against peers sharing a categorical:

```yaml
entities:
  - alias: facilities
    # …
    peer_groups:
      - by: facility_type          # required — the grouping categorical
        measures: [risk_score]     # optional — numeric columns to compare
```

(`peer_group: {by: …}` is sugar for a one-element list.) This synthesizes
`PEER_GROUP_SIZE`, `PEER_EVENT_RATE`, and per-measure `PEER_MEAN` /
`PEER_ZSCORE` / `PEER_PCTILE` / `EGO_MINUS_PEER_MEAN` features.

## Relationships

```yaml
relationships:
  - parent: {entity: customers, key: customer_id}
    child:  {entity: orders,    key: customer_id}
```

Backward traversal (parent ← child) applies **aggregations** over the child
rows per parent; forward traversal pulls parent attributes onto the child.
Parent and child keys may have different names — declare each side's own
column.

### Temporal (as-of) joins

```yaml
relationships:
  - parent: {entity: patients,   key: patient_id}
    child:  {entity: care_plans, key: patient_id}
    temporal:
      mode: as_of                # the only mode
      grace: P21D                # optional — look back at most this far
      child_timestamp: recorded  # optional — override the child's temporal_ix
```

Renders a `left join lateral … order by <timestamp> desc limit 1`: the most
recent child row at or before each `as_of_date` (bounded by `grace` when
given). This is the point-in-time join for slowly-changing state.

**Windows over a transferred value walk the target's timeline.** A window
transformer (`lag_*`, `rolling_*`, `cum_*`, `ema_*`, …) applied to a feature
brought in by a transfer — as-of or plain — orders by the *receiving* entity's
`temporal_ix`, not the source's. The transfer has already collapsed the
source's history to one row per receiving row, so the value is "the source's
state as of this row" and the only timeline left to walk is the receiver's.
The two windows mean different things and both are kept:
`CUM_SUM(plans.cost)` is computed on the source side and walks the source's
history; `CUM_SUM(plans.CUM_SUM(plans.cost))` is the same window applied again
after the transfer, and walks the receiver's rows.

**Known boundary (v1.0):** the correlated LATERAL cannot be flattened into
temp-table shards. If the entity carrying an as-of join *also* grows past
the oversized-CTE materialization threshold (issue-#7 sharding), featurizer
raises `NotImplementedError` — loudly, instead of emitting subtly-wrong
SQL. Workarounds: narrow that entity's primitive/interval breadth so its
synth stays under the limit, or attach the as-of relationship to the target
entity (the target is never shard-materialized). See the
[FAQ entry](/featurizer/faq/#cannot-yet-materialize-the-oversized-synth--as-of-lateral-join).

## Spatial relationships (planner pass)

With PostGIS and entities that declare a `spatial_ix`:

```yaml
spatial_relationships:
  - name: nearby           # feature-name token
    left: facilities       # target-side entity alias
    right: facilities      # other entity (self-joins allowed)
    within_m: 500          # colocation radius, meters
    bandwidth_m: 10000     # KDE bandwidth, meters
```

Synthesizes `COLOCATION_COUNT`, `DISTANCE_TO_NEAREST`, and `KDE_INTENSITY`
features between the two tables.

## Graph relationships (planner pass)

The taxonomy's cheap graph tier, in pure SQL — as-of degree and 1-hop
neighbour aggregates over an edge stream, no Python and no `[bridge]`
dependency:

```yaml
graph_relationships:
  - name: contacts             # required — feature-name token
    left: facilities           # required — node entity; features attach here
    right: states              # optional — neighbour-STATE entity (default: left)
    edges:                     # required — the edge stream
      table: contact_edges     # may be schema-qualified
      source: src_id           # edge source node-id column
      target: dst_id           # edge target node-id column
      timestamp: contacted_at  # REQUIRED — the pass is as-of by construction
    directed: true             # optional — false unions both directions
    measures: [risk_score]     # optional — default: right's numeric variables
    shares:   [flagged]        # optional — default: right's boolean variables
    features: [degree, neighbour_mean, neighbour_share]  # optional — default: all
```

Synthesizes, per `left` row:

- `DEGREE(<name>)` — edge incidences knowable as-of, plus one windowed
  `DEGREE(<name>|interval=P3M)` variant per configured interval;
- `NEIGHBOUR_MEAN(<name>.<measure>)` — mean of each numeric `measures`
  column over the 1-hop neighbours' state rows;
- `NEIGHBOUR_SHARE(<name>.<flag>)` — share of each boolean `shares` column
  (e.g. the flagged-neighbour rate).

Two causal bounds apply together: the **edge timestamp** and — when `right`
declares a `temporal_ix` — the **neighbour state's** timestamp, both cut at
the as-of date. That is why `edges.timestamp` is required: a static edge
table cannot be causally bounded. The pass is deliberately **1-hop only**;
2-hop aggregation pulls neighbours' future labels (the classic temporal-GNN
leakage) and is not offered. A neighbour reached by two knowable edges (or
carrying several knowable state rows) weighs proportionally in the
mean/share.

For heavier graph features (centralities, community membership) see the
[bridge cookbook](/featurizer/engineering/bridge-cookbook/) — including the
Path-2 move where near-duplicate or co-mention **text induces the edge
table** this block then consumes.

## Selecting primitives

```yaml
aggregations: [count, sum, mean, gap_cv, entropy]
transformations: [identity, abs, ln]
```

Both keys accept any registered name — browse the
[primitives reference](/featurizer/reference/primitives/) or run
`python -m featurizer list-primitives`. Omitting a key applies the full
default set; note that all defaults on a wide schema can synthesize past
PostgreSQL's 1664-column row limit, which featurizer handles by sharding the
output into column groups automatically (and warns when a config predicts a
pathological query plan).

## Column budget

### Each aggregation yields `I + 1` columns, not `I`

For every aggregated child column the planner calls each aggregator once with
no interval, then once per entry in `intervals`. With `I` intervals that is
`I + 1` columns: the lifetime value plus one per window.

```yaml
intervals: [P30D, P90D]      # I = 2
aggregations: [sum, mean]
```

renders, for a child column `orders.amount`, six columns and not four:

```text
SUM(orders.amount)     SUM(orders.amount|interval=P30D)     SUM(orders.amount|interval=P90D)
MEAN(orders.amount)    MEAN(orders.amount|interval=P30D)    MEAN(orders.amount|interval=P90D)
```

The lifetime column cannot be switched off. `intervals: []` still emits it,
and gives exactly one column per aggregation. The windowed variants are
skipped only for a child entity that declares no `temporal_ix`, because
there is nothing to window on. Budget the aggregation layer as

```text
child columns × aggregations × (I + 1)
```

and multiply by the transformers applied on top. At `I = 2` the lifetime
column is a third of the aggregation layer, which is easy to miss until you
count the output.

### The limit that binds a numeric matrix is row size, not column count

Three limits in this documentation are column counts: PostgreSQL's 1664
entries per target list, its 1600 columns per table, and featurizer's default
of 1400 columns per output group. A matrix of fixed-width numbers reaches a
fourth limit first. A PostgreSQL heap row must fit one 8 kB page (about
8160 usable bytes), and `double precision` and `bigint` are 8 bytes each and
cannot be moved out of line by TOAST. That puts the ceiling near **1,000
numeric columns per table**, under all three column counts. A 1,396-column
`float8` table fails with `row is too big: size 11176, maximum size 8160`.

`to_tables()` checks this before it writes. It estimates each group's row
width and re-partitions into narrower tables when a group would exceed an
8000-byte budget, which with two key columns allows 979 feature columns per
table. Reading the matrix with `to_dataframe()`, `to_arrow()` or
`to_parquet()` is not affected, because PostgreSQL streams a result row and
never stores it in a page.

**The check protects only the tables featurizer writes.** If you take
`Featurizer.query` and write the result yourself with `create table … as` or
`COPY`, nothing pre-flights the row width and PostgreSQL reports the error at
the first over-wide row. Split the write at the same 979 feature columns per
table, carrying `(as_of_date, <target id>)` in each, or call `to_tables()` and
read its groups. See the
[FAQ entry](/featurizer/faq/#row-is-too-big-size--maximum-size-8160--but-only-with-to_tables)
and [Staying under PostgreSQL's limits](/featurizer/engineering/internals/#staying-under-postgresqls-limits).

## Common validator messages

- `Unknown aggregation 'avg'` → *did you mean `mean`?* — names must match the
  registry exactly.
- `Invalid interval 'P1'` → intervals are full ISO-8601 durations (`P1D`,
  `P1M`, `P1Y`).
- `Target entity 'X' not found` → `target` must equal one entity's `alias`.
- `temporal block requires the child entity to declare temporal_ix (or
  child_timestamp)` — an as-of join needs a timestamp to order by.
