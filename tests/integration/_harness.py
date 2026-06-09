"""Helpers for executing a featurizer config against a live PostgreSQL.

The flow mirrors real usage: build the entity tables and an ``as_of_dates``
table as session ``TEMP`` tables on the caller's connection, render the config
to SQL via :class:`~featurizer.Featurizer`, execute it on the *same* connection
(so the temp tables resolve), and return the rows as plain dicts keyed by the
output column names.
"""

from __future__ import annotations

import tempfile
from typing import Any, Sequence

import yaml

from featurizer import Featurizer


def create_temp_table(
    conn: Any,
    name: str,
    columns: Sequence[tuple[str, str]],
    rows: Sequence[tuple[Any, ...]],
) -> None:
    """Create a TEMP table ``name`` with ``columns`` and insert ``rows``.

    Args:
        conn: An open psycopg connection (autocommit off).
        name: Table name (unqualified; resolved ahead of permanent tables).
        columns: ``[(column_name, sql_type), ...]``.
        rows: Tuples matching ``columns`` order.
    """
    cols_ddl = ", ".join(f"{col} {sqltype}" for col, sqltype in columns)
    with conn.cursor() as cur:
        cur.execute(f"create temp table {name} ({cols_ddl}) on commit drop")
        if rows:
            placeholders = ", ".join(["%s"] * len(columns))
            cur.executemany(f"insert into {name} values ({placeholders})", list(rows))


def run_featurizer(conn: Any, config: dict) -> list[dict[str, Any]]:
    """Render ``config`` to SQL, execute it on ``conn``, return rows as dicts."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    sql = Featurizer(path).query
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
