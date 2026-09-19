# featurizer task runner
#
# Integration workflow:
#   just db-up && just seed && just test-realistic && just db-down
#
# The test database is an ephemeral Docker container (no bind mounts, removed
# on stop). Application/test code never hardcodes credentials — it reads
# DATABASE_URL / PG* only; the URL below exists solely inside these recipes
# and points at the throwaway container.

set dotenv-load

pg_port   := "55432"
container := "featurizer-pg"
pg_url    := "postgresql://postgres:postgres@localhost:" + pg_port + "/featurizer_test"

default:
    @just --list

# Start an ephemeral PostgreSQL 16 for integration tests (removed on stop).
# Idempotent: several sessions share one machine, and `--rm` only removes the
# container when it STOPS, so a container another session left running used to
# fail this recipe on a name conflict. Reusing it is right — same image, name
# and port this recipe would have created — but note the corollary: the
# container is shared, so `just db-down` stops it for every session, not only
# yours.
db-up:
    @if [ -n "$(docker ps -q -f name=^{{container}}$)" ]; then \
      echo "reusing the running {{container}} on port {{pg_port}} (shared: db-down stops it for everyone)"; \
    else \
      docker run -d --rm --name {{container}} \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=featurizer_test \
        -p {{pg_port}}:5432 postgres:16; \
    fi
    @printf 'waiting for postgres'
    @until docker exec {{container}} pg_isready -U postgres -d featurizer_test >/dev/null 2>&1; do printf '.'; sleep 0.5; done
    @printf ' ready\n'

# Stop (and thereby remove) the test database container
db-down:
    docker stop {{container}}

# Download (cached under tests/data/) and load datasets into the test database
seed dataset="all":
    DATABASE_URL={{pg_url}} uv run python -m tests.integration.datasets seed {{dataset}}

# Full suite (integration tests skip unless a database is configured)
test:
    uv run pytest -q

# Fast tier only — no database needed
test-fast:
    uv run pytest -q -m "not integration"

# All integration tests against the ephemeral database
test-integration:
    DATABASE_URL={{pg_url}} uv run pytest -q -m integration

# Realistic-dataset tier only (requires `just seed` first)
test-realistic:
    DATABASE_URL={{pg_url}} uv run pytest -q -m "integration and slow"

# Seed + run ONE example end to end against the throwaway database
# (requires `just db-up`). NAME is a prefix: `just example 01` or
# `just example 04-custom-primitives`.
example NAME:
    dir=$(ls -d examples/{{NAME}}* | head -1); \
    DATABASE_URL={{pg_url}} uv run python "$dir/create_data.py"; \
    DATABASE_URL={{pg_url}} uv run python "$dir/run_example.py" --execute

# Seed + run ALL examples end to end (requires `just db-up`).
examples:
    just example 01
    just example 02
    just example 03
    just example 04
    just example 05
    just example 06

typecheck:
    uv run basedpyright

# Format the tree (ruff is pinned exactly in the dev group — see pyproject).
# Formatting ONLY: no `ruff check --fix` here. An autofix can be a semantic
# change — it once stood to delete the deliberate `cached_download` re-export
# in tests/integration/datasets/food_inspections.py, whose `# noqa: F401` sat
# on the wrong line — and that has no business riding along in a format recipe.
fmt:
    uv run ruff format .

# Formatting check only. CI gates on it (test.yml, the py 3.12 fast-tests job).
fmt-check:
    uv run ruff format --check .

# Advisory lint, not yet a CI gate. Two findings remain, both F541 in
# examples/02-temporal-joins/tutorial.ipynb, and they go away with the rewrite
# of that notebook (#25); after that this can gate too. Do not blanket `--fix`.
lint:
    uv run ruff check .

# Freeze v0.5.2 aggregator semantics as golden values (requires `just db-up`).
# Run ONCE before the set-based rewrite; never edit the JSON afterward.
bench-capture-golden:
    DATABASE_URL={{pg_url}} uv run python -m benchmarks capture-golden

# List the subquery aggregators + rewrite scope (no database needed).
bench-inventory:
    uv run python -m benchmarks inventory

# Report companion-CTE fan-out for a synthetic all-agg config (no database; plan P3).
bench-fanout:
    uv run python -m benchmarks.fanout_report

# Compare 63-byte truncation shapes (no database). Backs the won't-do in
# .out-of-scope/tail-preserving-truncation.md.
bench-truncation:
    uv run python -m benchmarks.truncation_shapes

# Scaling benchmark for the subquery-aggregator tier (requires `just db-up`).
# SCALE is one of 100 / 1k / 10k; artifacts land under
# specs/correlated-subquery-aggregator-scaling/.
bench-aggs SCALE="1k" TIMEOUT="300" LABEL="":
    DATABASE_URL={{pg_url}} uv run python -m benchmarks bench \
      --scale {{SCALE}} --timeout {{TIMEOUT}} {{ if LABEL != "" { "--label " + LABEL } else { "" } }}

# The last part of release step 1 in CONTRIBUTING.md:
#   just revendor-skill ~/.claude/skills/featurizer-dfs/SKILL.md <other copy>…
# Each copy keeps everything above its own H1 (its frontmatter and header
# comment); from the H1 down it becomes this repo's body. It refuses a copy
# that has no single H1 to anchor on, and ends by comparing the bodies byte for
# byte. It writes outside this repository: that is its job, so it takes the
# paths as arguments and commits none. Paths must not contain spaces.
# Re-vendor the featurizer-dfs skill body into other copies of the skill
revendor-skill +DESTS:
    #!/usr/bin/env bash
    set -euo pipefail
    src=".claude/skills/featurizer-dfs/SKILL.md"
    anchor='^# Featurizer'
    body_of() { awk -v re="$anchor" 'found || $0 ~ re { found = 1; print }' "$1"; }
    head_of() { awk -v re="$anchor" '$0 ~ re { exit } { print }' "$1"; }
    anchors_in() { grep -c "$anchor" "$1" || true; }
    fail() { echo "revendor-skill: $*" >&2; exit 1; }

    [ "$(anchors_in "$src")" = 1 ] \
      || fail "$src must have exactly one line matching '$anchor' (found $(anchors_in "$src")); the body starts there."
    for dest in {{DESTS}}; do
      [ -f "$dest" ] || fail "$dest is not a file. Pass the path of a vendored SKILL.md."
      [ "$(anchors_in "$dest")" = 1 ] \
        || fail "$dest has $(anchors_in "$dest") lines matching '$anchor', expected 1. Without that anchor there is no telling its header from its body; fix the copy by hand first."
      tmp="$(mktemp)"
      { head_of "$dest"; body_of "$src"; } > "$tmp"
      if cmp -s "$tmp" "$dest"; then
        rm -f "$tmp"
        echo "in parity already: $dest"
      else
        cat "$tmp" > "$dest"   # keeps the copy's permissions and any symlink
        rm -f "$tmp"
        echo "re-vendored:       $dest"
      fi
      cmp -s <(body_of "$src") <(body_of "$dest") \
        || fail "$dest still differs from $src below the H1 after writing it. Compare the two by hand."
    done
    echo "body parity holds across $(echo {{DESTS}} | wc -w | tr -d ' ') copies"
