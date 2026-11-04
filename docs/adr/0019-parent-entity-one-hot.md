# 0019 — One-hot for categoricals on parent entities reached by a forward relationship

**Status:** Proposed

**Date:** 2026-11-04

**Deciders:** Adolfo De Unánue

## Context

ADR-0007 deferred parent-entity one-hot, but master already encodes an unnamed
forward-transferred `role: categorical` with a declared vocabulary: the raw
variable reaches the target's feature set and is emitted as
`"<target_alias>.<column>=<value>"`, as-of correct, at depth 1 and depth 2. A
parent categorical whose vocabulary would come from its ENUM raises at
construction, one behind a named relationship passes through as a raw string
(`"<name>.<column>"`), and one that shares its column name with a target
categorical is silently dropped.

## Decision

Proposed, pending the maintainer: keep every name master emits today
byte-identical (the `<target_alias>` prefix included), and add only what fails
or is missing today: resolve the ENUM vocabulary of forward-transferred parent
categoricals at construction, so that a config that raises today runs. Renaming
to `"<parent_alias>.<column>=<value>"`, encoding named-relationship transfers,
and resolving the same-column collision change the output of configs that run
today, so each waits for an explicit naming decision.

## Consequences

No config that runs on master changes its output, so the change is additive
under ADR-0015 and triage-pg's parser sees no new name shape. The prefix of a
parent one-hot keeps naming the target rather than the entity that owns the
column, until the maintainer chooses a rename path (for example both names for
one minor release, with a deprecation warning, before the old one goes).
