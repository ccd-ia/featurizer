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
`uv sync --extra bridge`. The `[tui]` extra (the terminal cockpit,
`featurizer/tui/`, on lynkeus) installs on Python 3.12+ only — its
requirement carries a version marker, so on 3.10 and 3.11 `uv sync --extra
tui` installs nothing and `tests/test_tui*.py` skip themselves. `uv sync`
alone brings it on 3.12+ through the dev group.

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

## Stability & deprecation policy (v1.0+)

[ADR-0015](docs/adr/0015-v1-api-stability-commitment.md) defines what "stable"
means. The short form:

- **Frozen** (breaking = major version): the YAML config schema (incl. the
  `peer_groups` / `spatial_relationships` / `graph_relationships` planner-pass
  blocks), the `Featurizer` public surface (`query`, `query_groups`,
  `to_dataframe/arrow/parquet/tables`, `feature_manifest`,
  `manifest_dataframe` and their return shapes), the ADR-0007 output-naming
  contract (incl. 63-byte capping), the imputation contract, and the
  ADR-0001/0014 bridge contract (`compute` / `materialize*` / `emit_yaml`
  shapes, `persist=`, `model_vintage`).
- **Not frozen** (free to change in minors): planner/renderer internals, CTE
  names, generated SQL text, shard boundaries, module layout under
  `featurizer/primitives/`. The primitive set may *grow* in minors; removing
  or changing an existing primitive's values is breaking.
- **Semver**: breaking = major · additive = minor · fixes = patch.
- **Deprecations** warn via loguru (once per process) for **at least one
  minor release** before removal, and are listed in the CHANGELOG under the
  release that introduces the warning.

### Known, intentional carve-outs

These are deliberate and documented rather than silently configured — do not
"fix" them without an ADR:

- `pyrightconfig.json` ignores `featurizer/primitives/aggregations.py` and
  `featurizer/primitives/transformations.py`, and coverage excludes the same
  two modules. They are dynamic-variant heavy (dozens of generated primitive
  classes); their *real* coverage is the execution tiers — every registered
  primitive executes against live PostgreSQL in the integration suite, which
  asserts values, not just types.
- The coverage floor is **70%**, enforced on one designated CI leg (Python
  3.12). Raising it is welcome opportunistic work, not a release gate — the
  integration/realistic tiers carry the correctness burden the number
  doesn't show.

## Changing featurizer from a consumer or another session

`master` is protected. Nobody pushes to it — not a consumer session, not this
repository's own. Every change arrives as a pull request that goes green on the
eight required contexts (`fast tests (py 3.10 … 3.13)`, `integration
(PostgreSQL 14 · 16 · 17)`, `packaging (build + twine check)`), and the branch
must be current with `master` before it merges: rebase, `git push
--force-with-lease`, wait for the checks, then `gh pr merge --rebase
--delete-branch`. Linear history is required, so no merge commits.

Tags are not branches, and the release process below is unchanged by any of
this: `git push origin vX.Y.Z` still triggers `release.yml`. Put the tag on
`master` *after* the merge, on the merged commit — never on a pull-request
branch.

Six rules govern what may be a pull request at all.

**1. Issue first, and read before opening one.** Run `gh issue list` and `gh pr
list` before you open anything. If an issue or a pull request already covers
the need, comment there and pin its branch locally while you work — never open
a second one. One need, one thread.

**2. A need that touches every consumer is issue-only.** The Python floor
(3.10) or a new cap, `uv.lock`, anything on ADR-0015's freeze list (the YAML
config schema, the `Featurizer` public surface and its return shapes, the
ADR-0007 output-naming contract, the imputation contract, the ADR-0001/0014
bridge contract), an output-naming or imputation default: open an issue and
stop there. This repository's own session makes the change and cuts the
release, because it is the only session that can see all the consumers at
once. A freeze-list change is a **major version and a deprecation cycle**, as
the stability policy above and `CLAUDE.md` both say — the issue is where that
is decided, not a pull request that has already written the code.

**3. A local additive fix may be a pull request.** A new aggregation or
transformation primitive, a new φ-bridge family, a cockpit screen — additive,
nothing existing moved. The pull request references its issue and carries the
test that fails without it (the three-tier convention above says which tests a
new family owes). The review is three checks:

- CI green on all eight contexts;
- nothing on the freeze list moved;
- no signature changed.

**4. Pull requests do not edit `CHANGELOG.md`.** The release commit writes the
version's section from the merged pull requests; that section is what
`release.yml`'s guard already checks for at tag time (step 1 below). A pull
request describes its change in its own body instead — that body is the raw
material for the section. This is also why two open pull requests never
conflict on the same three lines under `## [Unreleased]`.

**5. Branch in a worktree off `origin/master`, named `<who>/<need>`.**

```bash
git worktree add .worktrees/<need> -b <who>/<need> origin/master
```

Another session may have `~/projects/featurizer` on a branch of its own; a
worktree off `origin/master` never moves under it. `.worktrees/` is gitignored.
Examples: `triage-pg/neighbour-share-denominator`, `policy/protected-master`.

**6. In a committed `pyproject.toml`, a dependency is pinned to a tag** — never
to a branch, a commit, or a path. Both directions of the seam obey this: the
`tui` extra pins lynkeus as `lynkeus @ git+…/lynkeus.git@vX.Y.Z ;
python_version >= '3.12'`, and consumers pin *this* repository the same way —
`triage-pg` carries `featurizer[parquet] @
git+https://github.com/ccd-ia/featurizer.git@v1.1.0`. There is no PyPI,
deliberately (see the release process below), so the tag **is** the
distribution channel and a branch pin is a moving target dressed as a version.
An editable path (`uv add --editable ../featurizer`) or a branch pin is a
working state: keep it in the uncommitted tree, swap it back to a tag before
the commit.

## Release process

Releases ride the CI/CD pipeline (`.github/workflows/release.yml`); nothing is
published by hand:

1. Add the `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` and bump
   `version` in `pyproject.toml` (then `uv lock`).
   In the same commit, update the `featurizer-dfs` Claude skill
   (`.claude/skills/featurizer-dfs/SKILL.md`): its heading version and pin
   line, and any surface the CHANGELOG section adds or changes.
   `tests/test_skill_parity.py` fails the fast tier when the skill disagrees
   with the code, so a forgotten update cannot reach the tag. Afterwards
   re-vendor the body (from the H1 down) into the other copies of the skill.
   Bump `version` and `date-released` in `CITATION.cff` in the same commit too
   — `tests/test_citation_parity.py` holds them to `pyproject.toml` and to the
   CHANGELOG section's date. That one matters more than it looks: a stale
   `version:` does not fail anything at tag time, it just mints a permanent DOI
   whose metadata names the wrong release.
2. Commit, push `master`, and wait for the `test` workflow to go green
   (fast + typecheck + packaging + example validation + integration).
3. Push an annotated tag: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`.
4. `release.yml` takes over: it fails loudly if the tag doesn't match
   `pyproject.toml` or the CHANGELOG section is missing, re-verifies the tagged
   commit, builds sdist+wheel, and creates the GitHub release with the
   CHANGELOG section as notes and the dist files as assets.
5. Zenodo archives the new release and mints a version DOI for it, on top of
   the permanent concept DOI that resolves to the latest version. Nothing to
   do per release — but bump `version` and `date-released` in `CITATION.cff`
   in step 1, because that is where Zenodo reads the record's metadata from.

No PyPI — deliberate (derived from dssg/featurizer; the name is generic).
GitHub releases on `ccd-ia/featurizer` are the distribution channel.

### Zenodo archiving

`CITATION.cff` is the **single** source of citation metadata. Do not add a
`.zenodo.json`: when a repository carries both, Zenodo uses the JSON and
ignores `CITATION.cff` entirely, which leaves two files to keep in sync and
demotes the CFF to the GitHub citation widget. Zenodo implements a subset of
the CFF schema — `cff-version`, `title`, `abstract`, `version`, `type`,
`license`, `message`, `authors` (with `orcid`) and `keywords` — so keep those
fields populated; anything else in the file is for human readers.

Zenodo only archives releases created **after** the repository is enabled in
its GitHub settings, so the toggle is a prerequisite for a tag, never a
follow-up to one.
