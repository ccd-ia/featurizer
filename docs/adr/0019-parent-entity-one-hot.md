# 0019 — One-hot for categoricals on parent entities reached by a forward relationship

**Status:** Accepted

**Date:** 2026-11-04

**Deciders:** Adolfo De Unánue

## Context

ADR-0007 deferred parent-entity one-hot, but master already encodes an unnamed
forward-transferred `role: categorical` with a declared vocabulary: the raw
variable reaches the target's feature set and is emitted as
`"<target_alias>.<column>=<value>"`, as-of correct, at depth 1 and depth 2. A
parent categorical whose vocabulary would come from its ENUM raises at
construction, one behind a named relationship passes through as a raw string
(`"<name>.<column>"`), and a transferred column of any type that shares its
name with a column the target already has is silently dropped (issue #82).

## Decision

Keep every name master emits today byte-identical, the `<target_alias>` prefix
included. Resolve the vocabulary of a forward-transferred parent categorical
the way the target's is resolved, from the declared list, else the ENUM of the
parent's own table, so a config that raises today runs. Report the name
collision with a warning instead of an error: the configs it affects run
today, and an error would stop them (ADR-0015). Renaming to
`"<parent_alias>.<column>=<value>"` and encoding named-relationship transfers
were rejected for 1.x, because each changes the output of configs that run
today.

## Consequences

No config that runs on master changes its output, so the change is additive
under ADR-0015 and triage-pg's parser sees no new name shape. The prefix of a
parent one-hot names the target, not the entity that owns the column, and a
parent categorical behind a named relationship stays a raw string column. A
rename belongs to 2.0. The planner reaches a database only through a resolver
the `Featurizer` passes in, and only for a parent categorical with no declared
vocabulary.
