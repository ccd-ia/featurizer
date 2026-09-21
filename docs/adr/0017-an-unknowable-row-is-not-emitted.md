# 0017 — A row that does not exist yet at a date is not emitted under it

**Status:** Accepted

**Date:** 2026-09-20

**Deciders:** Adolfo De Unánue

## Context

[ADR-0016](0016-leak-fixes-are-not-breaking.md) cut every *non-target* read on
the as-of date (issue #27) and left the target alone, on the ground that its
rows are the cohort. For a target with one row per entity and no temporal index
that holds, and there is nothing to cut on.

A target that declares a temporal index is different, and issue #49 measured
it on `8e1222e`. With one visit dated 2024-09-01 added to an event-like target
and the as-of date 2024-06-01:

- the later row was **emitted** under 2024-06-01;
- it moved `cross_entity_zscore` on 5 of the 5 knowable rows, `cusum` and `last`
  on 3 of 5, `percent_rank` on 2 of 5. `lag_*` and `cum_*` did not move.

The same cut reaches a target that is an *entity* table whose temporal index
says when the entity came into existence. Rows returned by the shipped
examples, before and after:

| example | target, temporal index | before | after |
|---|---|---|---|
| 01 | `customers.signup_date` | 1,200 | 844 |
| 02 | `patients.admission_date` | 400 | 273; the first as-of date returns none |
| 03 | `stores.open_date` | 160 | 129 |
| 05, 06, every `triage-pg` config | all rows before every date, or no temporal index | unchanged | unchanged |

The rows that go are the customer who signs up in August, listed under an
as-of date in January. Emitting that row tells a model trained as of January
that the customer will exist: cohort look-ahead. Their aggregates were already
NULL or zero, because #27 keeps the children's later rows out.

Alternatives considered:

1. **Keep every row and compute the whole-partition windows over the knowable
   rows only.** No returned row moves. An event-like target has several rows
   per id and no row key, so re-joining window values onto "all rows" needs a
   bounded companion of the target's synth and a way to match rows: new
   machinery in the part of the planner where the defects of 2026-09 came from,
   and the rows that should not be there stay.
2. **Opt-in in 1.x, default in 2.0.** Honours ADR-0015 to the letter and keeps
   the look-ahead as the default for a major. The maintainer rejected the same
   shape for #27.

## Decision

**A target that declares a temporal index is read with the same cut as every
other entity:** `<temporal_ix> <= aod.as_of_date` (`<` under the exclusive
boundary). A row dated after an as-of date is not emitted under it, and is
emitted from the first as-of date on or after its own.

This extends ADR-0016's exemption from *a value that depended on unknowable
rows* to *an unknowable row*. It ships in a minor. It is announced: the planner
logs a warning, once per process, for every target that declares a temporal
index, and the release notes lead with it.

A target **without** a temporal index is not cut, because there is no column to
cut on. That is how a caller asks for every target row under every date, and it
is what every `triage-pg` config does. Which of the rows that exist a date emits
is still the caller's decision, through `as_of_dates: {id_column}`.

## Consequences

- Row counts change for any config whose target declares a temporal index and
  has rows dated after an as-of date; an as-of date earlier than every row
  returns nothing. `len(matrix) == entities × dates` no longer holds for such a
  config.
- The four whole-partition transformer families are point-in-time correct at
  the target level; `tests/integration/test_asof_bounded_target_read.py` sweeps
  every numeric transformer on an event-like target (53 of 54 cases fail
  without the cut).
- **It overturns a behaviour the peer-group family had pinned.** Its tests
  emitted a future-born ego (an entity born in 2021, under a 2020 as-of date)
  and gave it the knowable members as peers: the pass already kept it out of
  everybody else's peer set, and still listed it. That ego is no longer
  emitted. Every value of the rows that remain is unchanged, and the two tests
  now say so (`tests/integration/test_realistic_peer_groups.py`).
- Under a paired cohort the two predicates combine with `and`. A
  population-level transformer under a paired cohort is filtered after the
  transform, as before, over a population that now holds only knowable rows.
- A consumer who relied on the dense grid has two ways back: drop `temporal_ix`
  from the target (as-of lookups *onto* the target then fall back to a static
  join, with the existing warning), or left-join the matrix onto their own
  entities × dates spine.
