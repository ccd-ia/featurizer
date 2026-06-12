"""Database connection for the seeding CLI.

Mirrors the rules of ``tests/integration/conftest.py``: the connection comes
from ``DATABASE_URL`` or the libpq ``PG*`` env vars, and there is no fallback
default — seeding into a guessed database is exactly the failure mode the
project's database hard-rule forbids.
"""

from __future__ import annotations

import os

import psycopg

_MISSING_ENV_MSG = (
    "No PostgreSQL configured: set DATABASE_URL or PG* env vars.\n"
    "For the ephemeral test database run:  just db-up && just seed"
)


def conninfo_from_env() -> str:
    """Return a libpq conninfo string from the environment, or exit loudly.

    Also used as the ``-d`` argument for ``pg_restore`` subprocesses.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if os.environ.get("PGDATABASE") or os.environ.get("PGHOST"):
        return ""  # empty conninfo: libpq reads PG* env vars
    raise SystemExit(_MISSING_ENV_MSG)


def connect_from_env() -> psycopg.Connection:
    """Open a psycopg connection (autocommit off; callers commit per dataset)."""
    conninfo = conninfo_from_env()
    try:
        return psycopg.connect(conninfo, autocommit=False)
    except psycopg.Error as exc:
        raise SystemExit(
            f"Could not connect to PostgreSQL ({conninfo or 'PG* env'}): {exc}\n"
            "Is the test database up?  just db-up"
        ) from exc
