# Tail-Preserving Identifier Truncation

**Decision:** Won't do — wrong fix. Rejected 2026-08-15.

**Reason:** It is a regression, not an improvement. Measured, it trades ~200
lost variable names for ~1,200 lost operator names and makes the very
glob-friendliness it was proposed to improve *worse*.

Featurizer caps generated feature names at PostgreSQL's 63-byte identifier
limit by keeping the **head**: `raw[:54] + "~" + md5(raw)[:8]`
(`featurizer/primitives/abstractions.py`, `_truncate_identifier`). On a deep
composition that erases the innermost — most informative — fragment:

```
CUM_SUM(facilities.MEAN(inspections.ABS(inspections.kw_rod~978dcf98
       └ full: CUM_SUM(facilities.MEAN(inspections.ABS(inspections.kw_rodent_complaint_flag)))
```

Anyone who meets that column proposes the same fix: keep the **tail** instead,
so `*(inspections.kw_*` matches. This file exists so that intuition gets
checked against the measurement rather than re-litigated.

## The measurement

`python -m benchmarks.truncation_shapes` (DB-free, reproducible) over
`featurizer/featurizer.yaml` — 2,104 columns, 1,217 of them truncating:

| scheme | variable lost | operator lost | ambiguous | worst group |
|---|---|---|---|---|
| head-keeping (today) | 197 | 0 | 887 | 8 |
| **tail-keeping (proposed)** | **0** | **1171** | **1027** | **78** |
| both-ends (26 + hash + 26) | 0 | 0 | 311 | 6 |

`ambiguous` counts columns whose visible, hash-elided text duplicates another
column's. No scheme has a correctness bug — the hash keeps every column unique
under all three. This is purely legibility.

## Why tail-keeping loses

It blinds you to the operator stack, which is exactly what distinguishes
sibling columns in a wide config. Its worst case is **78 columns** all
rendering as the same visible string:

```
…s.ABS(measurements.frecuencia_cardiaca)|interval=P2W))
   ↑ every ABS / PCT_CHANGE_1 / ROLLING_{MEAN_3,MEDIAN_7,STD_7,IQR_7} ×
     MAX / MEAN / MEDIAN / MIN / STDDEV / SUM collapsed into one
```

Head-keeping's worst group is 8. So the proposal does not make truncated names
glob-friendlier in general — it makes the *variable* glob work and breaks the
*operator* glob, on a config where the operator stack is what varies.

## Why even the good version isn't worth it

A both-ends shape (26 head + hash + 26 tail) genuinely **dominates** the status
quo: nothing lost at either end, ambiguity down 887 → 311, worst group 8 → 6.
If the truncation shape were ever going to change, that is the shape.

It still doesn't earn the change:

- **311 columns remain ambiguous.** You would still have to tell people to use
  the manifest, so the workaround the rename was meant to retire survives it.
- **Any shape change renames every currently-truncated column** — 1,217 of
  2,104 here, 57.8%. That silently invalidates every downstream feature cache
  and column-matching config, triage-pg's included.
- The naming contract *including 63-byte capping* is frozen under
  [ADR-0015](../docs/adr/0015-v1-api-stability-commitment.md), so it needs a
  major version and a deprecation cycle.

Paying a universal breaking rename to go from "physical names are unparseable"
to "physical names are slightly less unparseable" is a bad trade.

## The real conclusion

63 bytes cannot hold a compositional name, and no fixed-window truncation will
change that — every shape just picks which half to blind you to. The feature
manifest is not a consolation prize for a naming scheme we failed to get right;
it is the load-bearing design. Physical column names are **opaque handles**:
glob against `label`, then map to `column`.

The productive version of this request is therefore additive, not a rename:
make the manifest ergonomic enough that nobody reaches for physical names —
e.g. a `columns_matching("*(inspections.kw_*")` helper that globs labels and
returns physical columns. That is legal under the ADR-0015 freeze, ships in
1.x, costs nobody a cache invalidation, and also addresses the 887 columns that
are *already* ambiguous today and that no truncation change touches. Tracked in
`TODO.org`.

## If you want to reopen this

`tests/test_truncation_shapes.py` pins the ordering above (not the exact
counts, which move with the sample config's primitive set). Two of its
assertions are deliberately written as reopen triggers: if tail-keeping stops
trading away more operator names than it recovers variable names, or if
both-ends ever leaves physical names unambiguous, the test fails and this
decision is worth revisiting rather than the assertion worth relaxing.

## Prior requests

- 2026-08-07 — raised during the triage-pg v1.0.1 close-out (finding 2:
  "63-char truncation cuts mid-variable-name"). Evaluated and declined; the
  reasoning went into the FAQ (commit `6fb479d`), but with no measurement
  behind it and no tracker handle.
- 2026-08-15 — reopened, measured, and closed here. `TODO.org`'s standing
  "open a 2.x issue for tail-preserving truncation" item is cancelled by this
  file.
