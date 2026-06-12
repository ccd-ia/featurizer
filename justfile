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

typecheck:
    uv run basedpyright
