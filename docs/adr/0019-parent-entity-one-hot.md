# 0019 — One-hot for categoricals on parent entities reached by a forward relationship

**Status:** Proposed

**Date:** 2026-11-04

**Deciders:** Adolfo De Unánue

## Context

ADR-0007 deferred parent-entity one-hot, but releases up to 1.3 already encode
a `role: categorical` that an unnamed forward relationship carries into the
target, named after the *target* (`"<target_alias>.<column>=<value>"`), and
raise when its vocabulary would come from the parent's ENUM. Naming it after
the parent that owns the column renames columns that configs emit today, and
ADR-0007 freezes those names.

## Decision

A parent categorical reached by exactly one unnamed forward relationship at
depth 1 is emitted as `"<parent_alias>.<column>=<value>"` and also, for one
minor release, under its old `"<target_alias>.<column>=<value>"` name with a
once-per-process deprecation warning; the two families carry identical values,
as-of correct under `temporal: {mode: as_of}`. Its vocabulary is resolved like
a target's: the declared list, else the parent column's ENUM. Named
relationships, grandparents, several paths and a parent column shadowed by a
target column keep the output they had, pending their own naming decision.

## Consequences

A config that ran before keeps every column it had, with the same values, and
gains the parent-alias columns, so the change is additive under ADR-0015; the
minor release after the one that adds them removes the target-alias names,
which breaks a consumer still reading them. The one-hot `case` now delimits a
column name that is not a bare identifier, which only affects names that could
not run before; a reserved-word column name still fails, for target and parent
one-hots alike.
