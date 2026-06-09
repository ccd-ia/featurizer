"""Fixtures for the PostgreSQL integration harness.

These are the only tests that actually *execute* featurizer's generated SQL.
Everything else in the suite asserts on SQL fragments without running them, so
this harness is what catches dialect errors, column-resolution bugs, and value
miscalculations.

A connection is taken from ``DATABASE_URL`` or the standard libpq ``PG*``
environment variables (loaded by direnv in normal use). When neither is
configured the tests **skip** with a clear reason — they never fall back to a
guessed localhost database, per the project's database hard-rule.

Run with::

    uv run pytest -m integration
"""

from __future__ import annotations

import os

import pytest

try:  # psycopg is a runtime dependency, but guard anyway for a clean skip.
    import psycopg
except ImportError:  # pragma: no cover - psycopg is declared in pyproject
    psycopg = None  # type: ignore[assignment]


def _conninfo() -> str | None:
    """Return a libpq conninfo string, or None if no database is configured.

    - ``DATABASE_URL`` wins if set.
    - Otherwise, an empty conninfo lets libpq read ``PG*`` env vars, but only
      if at least ``PGDATABASE`` or ``PGHOST`` is present, so we never silently
      connect to a default localhost database.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if os.environ.get("PGDATABASE") or os.environ.get("PGHOST"):
        return ""
    return None


@pytest.fixture
def pg_conn():
    """A psycopg connection wrapped in a transaction that is always rolled back.

    Tests create ``TEMP`` tables on this connection; the rollback at teardown
    discards both the data and the temp tables, so the target database is left
    untouched.
    """
    if psycopg is None:
        pytest.skip("psycopg is not installed")
    conninfo = _conninfo()
    if conninfo is None:
        pytest.skip(
            "No PostgreSQL configured: set DATABASE_URL or PG* env vars "
            "(e.g. via direnv) to run the integration harness."
        )
    try:
        conn = psycopg.connect(conninfo, autocommit=False)
    except psycopg.Error as exc:  # surfaced in the skip reason, not swallowed
        pytest.skip(f"Could not connect to PostgreSQL ({conninfo or 'PG* env'}): {exc}")
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
