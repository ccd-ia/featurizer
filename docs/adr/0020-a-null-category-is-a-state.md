# 0020 — A NULL category is a state

**Status:** Proposed

**Date:** 2026-11-04

**Deciders:** Adolfo De Unánue

## Context

Issue #86. The sequence primitives build `(prev, curr)` pairs with `LAG(x)`,
and `LAG` returns NULL in two situations: the row has no previous row, or the
previous row's value is NULL. Five transition primitives and the two n-grams
filter `prev IS NOT NULL` (or `lag_1 IS NOT NULL`) to drop the first row, and
that filter also drops every step out of a NULL state. `rework_count` compares
with `prev = curr`, so a NULL never repeats. `ngram_3_freq` checks `lag_1` but
not `lag_2`, so it counts the second row of every series as a trigram.

Measured on PostgreSQL 16 on 2026-11-04, each value below is wrong under both
candidate rules, NULL as a state and NULL as missing data:

| sequence | primitive | today | NULL is a state | NULL is missing |
|---|---|---|---|---|
| `a, NULL, b` | transitions kept | `a→NULL` | `a→NULL, NULL→b` | `a→b` |
| `a, NULL, NULL, a` | `rework_count` | 0 | 1 | 1 |
| `a, NULL, NULL, a` | `state_volatility` | 1 | 2 | 0 |
| `a, b` | `ngram_3_freq` | 1 | NULL (no trigram) | NULL |

The engine already treats NULL as a state elsewhere, so the inconsistent
primitives disagree with their neighbours on the same rows. Of the 20
aggregations that take a categorical input (registry, 2026-11-04), by reading
the code (Phase 2 of the plan replaces this with measurements):

- **Values move (9):** eight are wrong under both rules: `sequence_entropy`,
  `markov_conditional_entropy`, `max_transition_prob`, `state_volatility`,
  `transition_matrix_summary`, `rework_count`, `ngram_2_freq` and
  `ngram_3_freq`. The ninth, `time_in_current_state`, already compares with
  `IS DISTINCT FROM`, but it does not count a NULL first row as a change, so
  an all-NULL window gives NULL. That one value is coherent under "NULL is
  missing" (see Decision).
- **Already a state (5):** `longest_streak`, `recurrence_interval`, `entropy`,
  `hhi`, `kl_drift`.
- **Predicates (3):** `first_passage_time`, `right_censoring_indicator`,
  `cross_type_latency`. A NULL matches no predicate, and a NULL row is still
  an event.
- **SQL aggregates (3):** `count`, `nunique`, `mode`, whose definitions ignore
  NULL inputs.

[ADR-0015](0015-v1-api-stability-commitment.md) says a change to emitted values
is breaking, and [ADR-0016](0016-leak-fixes-are-not-breaking.md) says a
different NULL rule is still breaking. The fix changes values.

## Decision

**A NULL is a state for every categorical aggregation that reads a value as a
category.** A primitive gives the same value on a series with NULLs as on the
same series with each NULL replaced by a category that appears nowhere else.

The SQL aggregates whose definition ignores NULL inputs, `count`, `nunique` and
`mode`, keep SQL semantics and are named as the exception. Changing them is a
semantic change outside #86.

**"No previous row" is a row number from the window that computes the LAG,
never the LAG's NULL.** `rn = 1` means there is no previous row; `rn > 1` with
a NULL `prev` means the previous state is NULL. The row number comes from the
same `WINDOW` as the `LAG` (one sort, the pattern `longest_streak` already uses
under [ADR-0018](0018-a-value-does-not-depend-on-the-physical-order-of-the-rows.md)),
and the change lands on the set-based pre-pass and the correlated fallback
together.

**It ships in a 1.x minor (1.5.0), without a deprecation cycle.** This extends
ADR-0016 and ADR-0018: a value that is wrong under every reading of the
primitive, under both NULL rules, was never part of the contract. The one
exception is a value that is coherent under another reading and still moves:
`time_in_current_state` on a window whose rows are all NULL, which is NULL
today and becomes the days since the window's first row under the rule. It is
named here, not hidden in the list. If the maintainer refuses it at signing,
that one case stays NULL and this ADR says so.

Alternatives considered:

- **NULL as missing data.** Also moves the five primitives that already treat
  NULL as a state, and `time_in_current_state` on every window that is not
  all NULL.
- **Opt-in in 1.x, default flipped in 2.0.** Keeps a wrong value as the default
  of a supported release for a major. Rejected before for #27, #49 and #66.
- **2.0 only.** The same, for longer.
- **A per-variable `nulls: skip` key.** Additive, so a minor could carry it.
  Recorded as possible future work and not built.

## Consequences

- **Moving cases:** to be filled in Phase 5 from the measured Phase 2 list,
  with the tests that show each value moved before and not after, and the
  render digest that shows nothing else moved (ADR-0016's three duties).
- **Measured on:** to be filled in Phase 5 from the Phase 4 runs on the live
  databases: moved cells, entities and columns per database, and NULL
  prevalence per categorical child column.
- An all-NULL series now has a defined dwell, entropy and streak. A downstream
  model that relied on NULL there will see numbers.
- The release notes name every primitive whose values move. No triage-pg
  config names any of the 11 sequence primitives (checked 2026-11-04), and the
  default aggregation selection contains none.
- A categorical primitive registered later is covered by the sentinel sweep the
  day it lands, which the sweep-matrix closure enforces.
