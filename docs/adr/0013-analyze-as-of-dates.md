# 0013 — Executor ANALYZEs the caller's `as_of_dates` before running

**Status:** Accepted
**Date:** 2026-07-09
**Deciders:** Adolfo De Unánue

## Context

Every generated query is `from as_of_dates cross join lateral (<features>)`. The
caller creates and populates `as_of_dates` (the set of point-in-time cuts); it is
typically a freshly-created / `TEMP` table that has **never been ANALYZEd**, so it
carries no statistics. PostgreSQL then assumes its hard-coded ~2550-row default,
and for a wide feature config it plans the lateral body for that wrong cardinality
— choosing a catastrophic join order/method.

Measured on the live triage DBs (`specs/reduce-child-stream-fanout-p3.html`):
an `EXPLAIN (ANALYZE)` of donorschoose all-agg was **99% one Merge Join at 277s**;
the top node estimated 7.65M rows (= 3000 entities × 2550 assumed as-of dates).
Running a single `ANALYZE as_of_dates` first collapsed it: **donorschoose 293.6s →
7.5s, dirtyduck (already drift-migrated) 27.6s → 7.0s** — ~40–50× on both, and
independent of the drift fix (ADR-0012).

## Decision

**The executor issues `ANALYZE as_of_dates` once, on its own working connection,
immediately before running the feature query** — in every execution path
(`to_dataframe` single-query + materialized, `to_arrow`, `to_tables`). It is:

- **Best-effort.** `ANALYZE` is a pure planner-stats optimization; a caller
  without ANALYZE privilege must still get correct (if slower) results, so a
  failure is logged (warning) and swallowed — never raised. This is not a hidden
  error: the query works regardless, and the log says the optimization was skipped.
- **Savepoint-isolated** on the psycopg paths, so a failed `ANALYZE` cannot poison
  the caller's open transaction (the integration harness / triage adapter pass a
  live, mid-transaction connection).
- **Read-only w.r.t. data.** `ANALYZE` only refreshes `pg_statistic`; it changes
  *plans*, never *values* — proven by the golden-value gate passing unchanged.

## Consequences

- **Positive:** wide full-aggregator configs became practical without any config
  change on the caller's side — both live wide DBs now materialize in ~7s. Accurate
  stats also help the *many*-as-of-date case (real temporal CV), not just the
  single-date case.
- **Surprising-without-context (why this ADR):** featurizer runs `ANALYZE` on a
  table it does not own. That is the point of recording it — it is deliberate,
  bounded (one table, best-effort, savepoint-isolated), and safe.
- **Not a substitute for** the still-open executor-side planner tuning
  (conservative `work_mem` / collapse limits, ~1.4× — see the perf spec); ANALYZE
  is the dominant lever, tuning is a smaller complementary one.
