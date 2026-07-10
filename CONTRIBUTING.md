# Contributing to Featurizer

Featurizer generates point-in-time-correct feature SQL for PostgreSQL. The
golden rule for every contribution: **a feature must never read the future**.
Each new family carries an explicit `<= aod.as_of_date` causal bound and is
verified against an independent recomputation.

## Setup

```bash
uv sync                      # create .venv and install deps (+ dev group)
uv run pytest -q             # fast tier (no database)
```

Use `uv run <tool>` so you get the locked versions. The optional `[viz]` and
`[bridge]` extras pull in heavy dependencies only when needed:
`uv sync --extra bridge`.

## Task runner

Prefer `just` recipes (run `just --list`):

- `just db-up` / `just db-down` — ephemeral PostgreSQL 16 in Docker (no bind
  mounts, removed on stop).
- `just seed` — load the realistic datasets into the test database.
- `just test-fast` — fast tier only (no database).
- `just test-integration` — all integration tests against the ephemeral DB.
- `just test-realistic` — the realistic-dataset tier (`integration and slow`).
- `just typecheck` — basedpyright (target: 0 errors).

## The three-tier test convention

Every feature family ships with all three tiers:

1. **DB-free shape guard** (`tests/test_planner_sql_validity.py`) — assert on the
   *shape* of the generated SQL (CTE present, causal bound carried, leave-one-out
   denominator, no token collisions). Catches regressions with no database.
2. **Inline PG value test** — run the family on a small synthetic fixture and
   compare exact values against hand-computed constants or an independent query.
3. **Realistic assertion** (`tests/integration/`) — run over a cohort of a seeded
   dataset and compare each value against an independent SQL/Python recomputation
   (`expect_sql`), asserting the causal cut directly. See the extension protocol
   in `tests/integration/_realistic.py`.

## Adding a primitive

Aggregations and transformations register via `register_aggregation` /
`register_transformer` (see `featurizer/primitives/`). Transformers must return a
**new** `Feature` (never mutate the input) to preserve hashing/dedup. Long
generated names go through `pg_identifier` (63-byte cap). Add the three tiers and
update the counts in `README` / `CLAUDE.md`.

## Adding a non-SQL family

If a feature can't be expressed as point-in-time-correct SQL, add a
`BridgeComputer` subclass in `featurizer/bridge/` (see ADR-0001/0003) and put its
dependency in the `[bridge]` extra — no engine change needed.

## Conventions

- Match surrounding style; `ruff`-clean, `basedpyright` 0 errors.
- Record hard-to-reverse, surprising, trade-off decisions as an ADR in
  `docs/adr/`; add domain terms to `CONTEXT.md`.
- Database access uses `DATABASE_URL` / `PG*` env only — never hardcode
  credentials.

## Release process

Releases ride the CI/CD pipeline (`.github/workflows/release.yml`); nothing is
published by hand:

1. Add the `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` and bump
   `version` in `pyproject.toml` (then `uv lock`).
2. Commit, push `master`, and wait for the `test` workflow to go green
   (fast + typecheck + packaging + example validation + integration).
3. Push an annotated tag: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
4. `release.yml` takes over: it fails loudly if the tag doesn't match
   `pyproject.toml` or the CHANGELOG section is missing, re-verifies the tagged
   commit, builds sdist+wheel, and creates the GitHub release with the
   CHANGELOG section as notes and the dist files as assets.

No PyPI — deliberate (derived from dssg/featurizer; the name is generic).
GitHub releases on `ccd-ia/featurizer` are the distribution channel.
