# 0011 — Domain-guarded ln / log / sqrt transformers

**Status:** Accepted
**Date:** 2026-07-09
**Deciders:** Adolfo De Unánue

## Context

`ln`, `log` (defined for x > 0) and `sqrt` (x >= 0) are unary math transformers.
Applied across a feature matrix they inevitably land on legitimately *signed*
inputs — z-scores, differences, deviations. PostgreSQL raises a hard error on
out-of-domain input (`cannot take logarithm of a negative number` / `... of
zero`), and because the whole matrix is one query, that single bad row aborts
the **entire** materialization. This surfaced on the live-DB `wide` variant
(all-agg × 14 transformers) against dirtyduck/chicago311 — see
`specs/feature-materialization-performance.html` (P4) and
`specs/live-db-revalidation-v060.html`.

Two options: (A) render SQL `NULL` for out-of-domain rows via `case when
<domain> then fn(x) end`; (B) restrict these transformers to non-negative inputs
only / make them opt-in.

## Decision

**Adopt Option A.** `ln`/`log`/`sqrt` are now `DomainGuardedTransformer`s whose
SQL is `case when x > 0 then ln(x) end` (`>= 0` for sqrt). Out-of-domain rows
become SQL `NULL` — an honest "undefined here" that a downstream imputer/encoder
already handles like any other NULL — instead of crashing the matrix.

This is **not** a violation of the project's fail-fast / never-swallow rule: a
non-positive argument to `ln` is a legitimate *domain condition* on real data,
not a bug or an unexpected error state. The alternative — losing 272 valid
features because one z-score was negative — is the worse failure mode. Only
`_build_transformer_call` changes; the output column `name`/`label` still derive
from `self.name`, so the ADR-0007 naming contract is byte-for-byte unchanged
(enforced by the transformer name/label tests and the collision snapshot).

## Consequences

- **Positive:** the `wide` / all-transformer configs no longer abort on signed
  features; math transformers are safe to include in a broad default set.
- **Negative:** an out-of-domain value is now a silent `NULL` rather than a loud
  error; a caller who *wanted* the crash (to detect that they applied `ln` to a
  signed feature) no longer gets it. Mitigated by the manifest/description making
  the transformer explicit, and by `NULL`s being visible in the output.
- **Scope:** only the real-domain unary transformers are guarded. `cbrt`, `exp`,
  `abs`, `sign` etc. are total functions and are unchanged. Aggregators that use
  `ln` internally (geometric_mean, theil, entropy) already guard their own inputs
  and are out of scope.
