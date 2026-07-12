---
title: Configuration reference
description: >-
  Every key of featurizer's YAML configuration — entities, variables and
  roles, relationships, temporal joins, intervals, primitive selection,
  peer groups and spatial features.
sidebar:
  order: 2
---

One YAML file describes your schema and what to synthesize. The annotated
skeleton, then every key in detail:

```yaml
target: customers            # required — entity alias features are FOR
max_depth: 2                 # required — relationship traversal depth
intervals: [P7D, P30D]       # required — ISO-8601 rolling windows

aggregations: [count, mean]  # optional — default: all 67
transformations: [identity]  # optional — default: all 83
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
| `intervals` | yes | ISO-8601 durations (`P7D`, `P1M`, `P1Y`…); every interval multiplies the windowed aggregations |
| `aggregations` | no | subset of registered aggregations to apply — see the [primitives reference](/featurizer/reference/primitives/); omit for the full default set |
| `transformations` | no | subset of registered transformers; omit for the full default set |
| `as_of_boundary` | no | `inclusive` (events at the as-of date count) or `exclusive` (strictly before) |
| `entities` | yes | the tables — see below |
| `relationships` | no | foreign-key links — see below |
| `spatial_relationships` | no | second-table spatial features — see below |

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

## Common validator messages

- `Unknown aggregation 'avg'` → *did you mean `mean`?* — names must match the
  registry exactly.
- `Invalid interval 'P1'` → intervals are full ISO-8601 durations (`P1D`,
  `P1M`, `P1Y`).
- `Target entity 'X' not found` → `target` must equal one entity's `alias`.
- `temporal block requires the child entity to declare temporal_ix (or
  child_timestamp)` — an as-of join needs a timestamp to order by.
