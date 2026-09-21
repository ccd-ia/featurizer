# 0016 — A point-in-time leak fix is not a breaking change

**Status:** Accepted

**Date:** 2026-09-19

**Deciders:** Adolfo De Unánue

## Context

[ADR-0015](0015-v1-api-stability-commitment.md) says a change to an existing
primitive's emitted values is breaking: a major version and a deprecation
cycle. Issue #27 then measured six transformers whose value on a child entity
moved when a child row dated *after* the as-of date was added — `percent_rank`,
`ntile`, `last`, `cusum`, `cross_entity_zscore`, `cross_entity_percentile` —
and a seventh, `cdf`, that would once it ran. The aggregation cut the child on
the as-of date, but only after the transformers had windowed over the whole
child table, and a window over its whole partition carries what it saw.

Every fix changes what those primitives emit. Read literally, ADR-0015 sends
the fix to 2.0.0 and leaves a known leak on every 1.x release until then.

The alternatives were to ship the fix behind an opt-in key and flip the default
in 2.0, or to hold it for 2.0 outright. Both keep the leak as the default
behaviour of a supported release.

## Decision

**The freeze protects point-in-time-correct values.** A value that depended on
rows dated after the as-of date was never part of the contract, because
point-in-time correctness is the invariant the contract exists to serve. A
change whose only effect on emitted values is to remove that dependence ships
in a **minor** release, without a deprecation cycle.

Such a change must:

1. carry a test that adds a row dated after the as-of date and shows the value
   moved before the change and does not after it;
2. show that no value moves for a primitive that did not read the future;
3. name, in the CHANGELOG entry of the release, every primitive whose values
   move.

[ADR-0017](0017-an-unknowable-row-is-not-emitted.md) later extended this from
a value to a row: an unknowable *row* is not emitted either.

Everything else in ADR-0015 stands. A value change for any other reason — a
different definition, a different default, a different NULL rule — is still
breaking.

## Consequences

- #27 ships in 1.x: a non-target entity's read is cut on the as-of date where
  it is read (`FeaturePlanner._causal_where`), and `cdf` is switched on. The
  seven primitives above change value for an entity that has rows after an
  as-of date; backward-only windows (`lag_*`, `cum_*`, `rolling_*`, `ema_*`)
  cannot move, and `tests/integration/test_asof_bounded_child_read.py` sweeps
  the whole registry to show it.
- A consumer who trained on one of the seven gets different, correct, features
  after the upgrade, and has to retrain. None is in the curated default set and
  no triage-pg config names one; the release notes say so anyway.
- The exemption is narrow by construction. "It was a bug" is not enough: the
  test in (1) is what qualifies a change, and only a dependence on unknowable
  rows can make that test fail before and succeed after.
