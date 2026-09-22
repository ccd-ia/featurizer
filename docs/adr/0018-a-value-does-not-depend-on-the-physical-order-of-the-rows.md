# 0018 — A value does not depend on the physical order of the rows

**Status:** Accepted

**Date:** 2026-09-21

**Deciders:** Adolfo De Unánue

## Context

A primitive that walks a timeline orders its rows by the entity's temporal
index: `lag(x) over (partition by <key> order by <temporal_ix>)`, and the same
`order by` under every transition, run length, autocorrelation, difference and
rolling window. Two rows of one partition on the same timestamp have no order
under that clause, so PostgreSQL returns them as they were read, which is the
physical order of the table.

Issue #66 measured it on the plain dense query, master `9fd2a8b`, with the
same child rows stored in another physical order (a TEMP copy written `order by
random()`):

| database | child | entities | with two child rows on one timestamp | columns that moved | entities that moved |
|---|---|---|---|---|---|
| dirtyduck | `inspections` | 22,169 | 106 | 28 | 25, all 25 with a tie |
| donorschoose | `resources` | 3,000 | 1,962 | 36 | 1,437, all with a tie |

A sweep over both registries with two rows per series on one date: 11 of 67
aggregations and 28 of 83 transformers return a different value when the rows
are inserted backwards, and so does an as-of lookup whose source has two rows
on one timestamp (`limit 1` under `order by <temporal_ix> desc`). The value is
stable while the scan order is; a reload, a `cluster`, a `vacuum full` or a
different plan above the scan changes it. A `date` temporal index makes ties
the normal case, not an edge: every event of a day shares one.

Issue #67 is the same family of defect in one primitive.
`cosinor_amplitude_weekly` regresses the value on the weekly sine and cosine of
the timestamp; rows a whole number of weeks apart share a phase, the basis is
constant up to the rounding of `sin()` at an argument near 1e4 radians, and
`regr_slope` divides by that rounding. Prices in the hundreds gave amplitudes
of 1e15, and a different one on every read.

[ADR-0015](0015-v1-api-stability-commitment.md) says a change to a primitive's
emitted values is breaking. Both fixes change values: every tied partition, and
every series on one phase.

## Decision

**A value that depended on the physical order of the rows, or on rounding
noise, was never reproducible, so it was never part of the contract.** Making
it reproducible ships in a **minor** release, without a deprecation cycle, on
the reasoning of [ADR-0016](0016-leak-fixes-are-not-breaking.md): the freeze
protects values a consumer could have relied on, and these could not be
relied on twice.

Two rules follow, and both are enforced by registry-wide sweeps:

1. **Every entity has one row order, and every window uses it.** After the
   temporal index a window orders by the entity's other identifier columns
   (its id first, then relationship keys and carried index variables, minus
   the column the window partitions by), then by every declared variable a
   window can compare (`Entity.tiebreak_columns`; not `index`-typed variables,
   which are identifiers already, not `vector`, and not `role: identifier`,
   which the target's synth does not carry). Two rows that still tie are the
   same base row twice, and every value the engine derives from them is the
   same. One list for every window over the entity, whatever column it reads,
   so they share one sort. The rolling percentiles' "up to and including the
   current row" becomes a row comparison over the same columns; the as-of
   lookup's `limit 1` takes the last row in that order, every column
   descending. It applies to aggregations (`_timeline` in `aggregations.py`,
   the set-based pre-pass and the correlated subquery of each family),
   transformers (`_ordering_columns` in `transformations.py`, which
   `_temporal_ordering` renders) and the planner's as-of lateral. Sweep:
   `tests/integration/test_tied_timestamps.py`, every registered primitive and
   the lookup, the same rows in two physical orders, equal values.
2. **A reduction whose basis has no spread returns NULL.** `cosinor_amplitude_*`
   returns NULL unless both basis columns have a variance above 1e-12 (rounding
   noise is near 1e-24; two rows a second apart on a weekly period are already
   at 2.5e-11; a real cycle is between 0.1 and 0.5). Sweep:
   `tests/integration/test_degenerate_phase.py`, every numeric aggregation on a
   series whose timestamps share the weekly phase returns NULL or a number of
   the data's size.

Alternatives considered:

- **Opt-in in 1.x, default in 2.0.** Keeps an irreproducible value as the
  default of a supported release for a major. Rejected for the same reason the
  maintainer rejected it for #27 and #49.
- **Document only: make the temporal index unique within a key.** Not possible
  for a `date` index on an event table, which is most of them.
- **Tiebreak by the identifier columns and the column the primitive reads.**
  Built first. Two rows equal in those but different in another variable
  still traded the values they received, and the row a consumer tells apart
  by that other variable got either. Ordering by every declared variable
  leaves nothing to trade. It also gives every window of an entity the same
  `order by`, one sort instead of one per input column: the live dirtyduck
  wide cell went from 7.5 s to 6.3 s.
- **The run-length family's two ranks.** `longest_streak` numbered the rows
  twice, once per key and once per key and value, and subtracted; the two
  sorts could order two identical rows differently and split a run. It now
  numbers the rows once and derives the run id from that numbering.

## Consequences

- **Values move only where there was a tie or a degenerate basis.** On
  dirtyduck 25 of 22,169 entities, on donorschoose 1,437 of 3,000; the release
  notes name every primitive that can move: the 11 aggregations and 28
  transformers of the sweep, an as-of lookup with tied source rows, and
  `cosinor_amplitude_weekly`. With the fix the same dense query over a
  shuffled copy of the child moves 0 columns on both databases, and a paired
  cohort equals the dense run on every pair of every cell measured.
- Rendered SQL changes for every config that selects a window transformer, an
  order-dependent aggregation or an as-of lookup: each `order by` gains
  columns, and a column-group synth gains the declared variables its window
  now names (the literal-name rule of the sharder). Nothing else moves:
  `tests/fixtures/render_baseline_pre_cohort.json` is re-captured with
  `ordered_by: "#66"` and its lineage note has the line counts. CTE names,
  transform select lists and output names do not move.
- The sweep matrix gains its seventh invariant, *a value does not depend on
  the physical order of the rows*, enforced registry-wide by
  `tests/test_sweep_matrix_coverage.py`. The next order-dependent primitive
  somebody registers meets a tie the day it lands.
- A window now reads the entity's identifier columns and declared variables,
  which the synth already projects (a column-group synth keeps the ones a
  window names); the transform CTE's select list is unchanged.
