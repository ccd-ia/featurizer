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

# Start an ephemeral PostgreSQL 16 for integration tests (removed on stop)
db-up:
    docker run -d --rm --name {{container}} \
      -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=featurizer_test \
      -p {{pg_port}}:5432 postgres:16
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
# Formatting ONLY: no `ruff check --fix` here. Autofix would delete the
# deliberate `cached_download` re-export in tests/integration/datasets/
# food_inspections.py (its `# noqa: F401` sits on the closing paren, one line
# below the diagnostic, so it never applies) — a semantic change has no
# business riding along in a format recipe.
fmt:
    uv run ruff format .

# Formatting check only — safe to gate CI on, the tree passes it today.
fmt-check:
    uv run ruff format --check .

# Advisory lint. NOT clean: 8 pre-existing findings (4 F401 unused imports,
# 4 F541 placeholder-less f-strings in example notebooks). Cleaning them is a
# separate change from adopting the formatter — do not blanket `--fix`.
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

# Scaling benchmark for the subquery-aggregator tier (requires `just db-up`).
# SCALE is one of 100 / 1k / 10k; artifacts land under
# specs/correlated-subquery-aggregator-scaling/.
bench-aggs SCALE="1k" TIMEOUT="300" LABEL="":
    DATABASE_URL={{pg_url}} uv run python -m benchmarks bench \
      --scale {{SCALE}} --timeout {{TIMEOUT}} {{ if LABEL != "" { "--label " + LABEL } else { "" } }}
