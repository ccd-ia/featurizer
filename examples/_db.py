"""Shared PostgreSQL helpers for the examples.

Featurizer emits PostgreSQL-dialect SQL (lateral joins, ``percentile_cont …
within group``, ``::date`` casts), so the examples execute against a real
PostgreSQL — not SQLite. The connection is read from the environment
(``DATABASE_URL`` or ``PG*``), per the project's database hard-rule; it is never
hardcoded. The quickest way to get one is the throwaway container::

    just db-up            # starts postgres:16 and sets DATABASE_URL in the recipe
    just example 01       # seed + run a single example end to end
    just db-down          # tear it down

or point ``DATABASE_URL`` / ``PG*`` at any PostgreSQL you already have.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import quote

import psycopg


def conninfo_from_env() -> str | None:
    """A libpq conninfo / URL from the environment, or ``None`` if unconfigured.

    ``DATABASE_URL`` wins. Otherwise an empty conninfo lets libpq read ``PG*``
    env vars, but only when ``PGDATABASE`` or ``PGHOST`` is present — so we never
    silently connect to a default localhost database.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if os.environ.get("PGDATABASE") or os.environ.get("PGHOST"):
        return ""
    return None


def require_conninfo() -> str:
    """:func:`conninfo_from_env`, or exit with actionable guidance."""
    info = conninfo_from_env()
    if info is None:
        sys.exit(
            "No PostgreSQL configured. Start the throwaway database with "
            "`just db-up` (the recipe exports DATABASE_URL), or export "
            "DATABASE_URL / PG* to point at your own PostgreSQL."
        )
    return info


def connect(schema: str) -> psycopg.Connection:
    """Open a connection, (re)create ``schema``, and set it as the search_path.

    Returns an open psycopg connection inside a transaction; the caller creates
    tables / inserts rows with **bare** names (resolved via the search_path) and
    calls ``commit()``. Re-running drops the prior schema, so seeding is
    idempotent.
    """
    conn = psycopg.connect(require_conninfo())
    with conn.cursor() as cur:
        cur.execute(f'drop schema if exists "{schema}" cascade')
        cur.execute(f'create schema "{schema}"')
        cur.execute(f'set search_path to "{schema}"')
    return conn


def _base_url() -> str:
    """A ``postgresql://`` URL from the environment (DATABASE_URL or PG*)."""
    info = require_conninfo()
    if info != "":
        return info
    # PG* path: assemble a URL SQLAlchemy can parse.
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    db = os.environ["PGDATABASE"]
    user = os.environ.get("PGUSER", "")
    password = os.environ.get("PGPASSWORD", "")
    auth = f"{quote(user)}:{quote(password)}@" if user else ""
    return f"postgresql://{auth}{host}:{port}/{db}"


def records_url(schema: str) -> str:
    """A ``DATABASE_URL`` for ``records``/SQLAlchemy that targets ``schema``.

    Two adjustments over the raw env URL:

    - **Force psycopg3.** ``records.Database`` defaults a bare ``postgresql://``
      URL to psycopg2, which this project does not depend on; normalize the
      scheme to the explicit ``postgresql+psycopg://`` driver.
    - **Pin the search_path** via a libpq ``options=-csearch_path=<schema>`` so
      the generated SQL's bare table names (and ``as_of_dates``) resolve.
    """
    base = _base_url()
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if base.startswith(prefix):
            base = "postgresql+psycopg://" + base[len(prefix) :]
            break
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}options={quote(f'-csearch_path={schema}')}"
