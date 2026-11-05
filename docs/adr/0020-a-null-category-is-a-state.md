# 0020 — A NULL category is a state

**Status:** Accepted

**Date:** 2026-11-05 (proposed 2026-11-04)

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
aggregations that take a categorical input (registry, 2026-11-04), as measured
by the sweeps in `tests/integration/test_null_state.py` on 2026-11-05 (the
reading of the code had predicted the same nine):

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
named here, not hidden in the list, and the maintainer accepted it at signing
on 2026-11-05: one rule, no special case.

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

- **Moving cases**, following ADR-0016's three duties:
  - *Each value moved before and not after.* Two oracles in
    `tests/integration/test_null_state.py`. The sentinel sweep compares 17
    categorical aggregations (20 minus the three SQL aggregates, each of which
    a test shows does move) with the same rows where each NULL is replaced by
    an unused category. The reference sweep compares the 11 sequence
    primitives on both paths with the Python reference in
    `tests/_sequence_reference.py`. A DB-free guard,
    `tests/primitives/test_sequence_null_marker.py`, keeps the rendered SQL
    from reading a LAG's NULL. Before the fix, 9 sentinel, 18 reference and
    18 guard cases failed, the nine primitives above on both paths; after it,
    all 63 pass. Reverting the engine fails the same 45.
  - *Nothing else moved.* A per-name render digest of all 68 registered
    aggregations is byte-identical for 59 of them; the 9 that moved are the
    nine above. The fallback parity sweep stays 30/30, and the tied-row and
    future-row sweeps are unchanged.
  - *Every moving primitive named:* the nine above, for the 1.5.0 CHANGELOG.
    `ngram_3_freq` also moves on series with no NULL at all. Under the old
    `lag_1 IS NOT NULL` test, a two-row series counted a trigram of 1, and
    under the rule it has none (NULL). On real data this is most of what
    moves.
  - The frozen v0.5.2 golden (`tests/fixtures/preagg_golden_values.json`)
    held the partial trigram for the three two-row groups of its edge
    fixture. The test overrides those six cells by name and leaves the file
    as captured.
- **One fallback defect fixed on the way.** The reference sweep found that
  the correlated fallbacks of `state_volatility`, `transition_matrix_summary`
  and `rework_count` counted 0 on an empty window of a key with history,
  where the set-based pre-pass, the path the planner renders, gives NULL. The
  fallback now gives NULL too. No rendered value changes because of it.
- **Measured on** the three triage-pg databases (2026-11-05, PostgreSQL 16,
  one as-of date each: the latest knowledge date). The configs are the triage
  entity graphs with the 11 sequence primitives;
  `benchmarks/null_state_impact.py` was run on `master` and on the fix. The
  control, two independent `master` runs, differed in 0 cells on each
  database.

  | database | rows × columns | moved cells | entities | columns | NULLs in the categorical child |
  |---|---|---|---|---|---|
  | dirtyduck | 22,169 × 133 | 16,781 | 4,452 | 18 | `inspections.risk` 4 of 74,191; `result`, `type` 0 |
  | donorschoose | 3,000 × 44 | 932 | 464 | 4 | `resources.project_resource_type` 0 of 16,682 |
  | chicago311 | 30,654 × 44 | 0 | 0 | 0 | `area_backlog.sr_type` 0 of 30,654 |

  Of dirtyduck's 16,781 cells, 16,773 are `ngram_3_freq` partial trigrams
  (1 to NULL). The other 8 are on the 4 entities that hold the 4 NULL `risk`
  rows, and 3 of those are one-row all-NULL dwells. All 932 donorschoose
  cells are partial trigrams.
- An all-NULL series now has a defined dwell, entropy and streak. A downstream
  model that relied on NULL there will see numbers.
- The release notes name every primitive whose values move. No triage-pg
  config names any of the 11 sequence primitives (re-checked 2026-11-05), and
  the default aggregation selection contains none.
- A categorical primitive registered later is covered by the sentinel sweep the
  day it lands, which the sweep-matrix closure enforces.
