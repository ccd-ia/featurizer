# 0021 — An interval window cuts at the as-of instant

**Status:** Proposed

**Date:** 2026-11-05

**Deciders:** Adolfo De Unánue

## Context

Issue #87. The engine spelled the as-of boundary two ways. The read of every
dated entity is cut with `ts <= aod.as_of_date` (`<` under `exclusive`), which
compares instants: a `date` as-of date is 00:00 of its day. The interval
windows were `daterange((aod.as_of_date - interval 'W')::date,
aod.as_of_date::date, '[]') @> ts::date`, which compares calendar days. On a
`date` column with a `date` as-of date the two select the same rows. On a
`timestamp` they do not, and three defects followed, measured on PostgreSQL
16.14 with the issue's five rows (`tests/integration/test_timestamp_as_of.py`):

| shape | column | before | instant |
|---|---|---|---|
| `exclusive`, as-of `timestamp '2024-06-01 18:00'` | `COUNT` whole / P1D / P30D | 3 / 1 / 1 | 3 / 2 / 3 |
| `inclusive`, same as-of date | `COUNT` whole / P1D / P30D | 3 / 3 / 3 | 3 / 2 / 3 |
| either mode, as-of `date '2024-06-01'` | any | unchanged | unchanged |

1. Under `exclusive` with a `timestamp` as-of date, the window's open upper
   bound at the as-of *day* dropped every row of that day, rows the read cut
   kept. A P30D column missed two rows from inside its own 30 days.
2. The `::date` cast truncated the lower bound to a day: on a `timestamp`
   as-of date, P1D reached back 42 hours.
3. The documentation described the as-of date in calendar-day words ("a row
   dated on the as-of date is knowable"); the engine reads it as an instant.

Nothing leaked: every window reads rows the causal cut has already bounded.
The window's own upper bound would have admitted the rest of the as-of day
had a future pass applied it to an uncut read.

The sweep written for this issue, which compares each interval column with the
same query over only the rows inside its window, found a fourth defect that has
nothing to do with timestamps. `skewness`, `kurtosis`, `harmonic_mean` and
`geometric_mean` build their expression from several aggregate calls and
override the method that adds the window filter. They took `interval` and
applied none. Every `|interval=W` column of theirs, on any column type and any
as-of date, has held the whole-history value since the primitives existed. All
four are in the default set of 65.

## Decision

**An interval window is written on the column at both bounds, and its upper
bound is the causal predicate**: `(ts >= aod.as_of_date - interval 'W' and ts
<= aod.as_of_date)` (`<` under `exclusive`), from one helper,
`boundary.interval_window`. A window cannot hold a row the whole history does
not. The two-window drift baseline is `(ts >= aod.as_of_date - 2W and ts <
aod.as_of_date - W)`, its upper bound open whatever the mode, as before.

**The engine supports a `timestamp` as-of date, and its window follows its instant**:
P1D is the 24 hours before it. The alternative, rejecting a non-`date` as-of
date, breaks a shape that runs today and moves the question to 2.0 without
answering it.

**The as-of date is an instant, and the docs say so**: a `date` is 00:00 of its
day, so on a `timestamp` column a row stamped later that day is knowable from
the next as-of date. `timestamptz` values on either side follow the session
`TimeZone`.

**The four aggregators filter every aggregate call they make**, through
`_window_filter`. The filter goes on each call, not into a `CASE` around the
column: `geometric_mean`'s domain guard maps a NULL to 1, so a column nulled
outside the window would be averaged in as `ln 1 = 0`.

**Unchanged on purpose**: `recency`, `tenure`, `inter_event_hazard_proxy` and
the other primitives that subtract dates still count calendar days
(`aod.as_of_date::date - ts::date`); their unit is days. The bridge's Python
comparisons (`featurizer/bridge/base.py`) are not touched.

**It ships in 1.5.0, a minor, without a deprecation cycle.** ADR-0015 freezes
emitted values; ADR-0016 exempts leak fixes, and this is not one. The case for
a minor is the one ADR-0020 made: each move makes a column hold what its frozen
name says. `AGG(x|interval=W)` names a window; the four aggregators ignored it,
and the timestamp shapes measured a different window from the one named.

## Consequences

What moves:

- Every `|interval=W` column of `skewness`, `kurtosis`, `harmonic_mean` and
  `geometric_mean`, on every config that selects them, the default set
  included. This is the largest move. It is not measured on the live
  databases yet.
- On a `timestamp` (or `timestamptz`) as-of date: the interval columns of
  every aggregation, in both modes.
- An interval with a time part (`PT12H`, `P1DT6H`) on any as-of date: its
  lower bound is no longer rounded down to a day. No shipped config, test or
  document uses one.

What does not: a `date` as-of date with intervals of whole days, weeks,
months or years, on a `date`, `timestamp` or `timestamptz` column. The render
baseline moved in 13 of 23 parts, and each is the old text once the windows are
rewritten in the new form (`tests/utils/render_baseline.py`). The integration
tier, with its frozen goldens, passed on the window change alone, before the
fix to the four aggregators.

`tests/integration/test_timestamp_as_of.py` holds the sweep: on a `timestamp`
child with a `timestamp` as-of date, in both modes, for every registered
aggregation, a row outside the window moves no interval column, and a window
holding every row equals the whole history. On `a18bff9` it fails 113 and 56
cases respectively, and the two `timestamp` hand cases; with this decision it
passes.
