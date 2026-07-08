"""Database plumbing for the benchmark + golden-capture tooling.

Connection resolution matches the integration harness (``DATABASE_URL`` or
``PG*``; never a guessed localhost) and fails loud when nothing is configured,
per the project database hard-rule. Fixture seeding and config execution are
reimplemented here (not imported from ``tests``) so this package stays
independent of the test tree.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from featurizer import Featurizer

from . import preagg_cases

try:  # psycopg is a runtime dependency; guard for a clean error message anyway.
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


def _conninfo() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if os.environ.get("PGDATABASE") or os.environ.get("PGHOST"):
        return ""
    raise EnvironmentError(
        "No PostgreSQL configured for benchmarks: set DATABASE_URL or PG* env "
        "vars (e.g. `just db-up` exports DATABASE_URL). Refusing to guess a "
        "localhost database."
    )


def connect() -> Any:
    """Open a psycopg connection (autocommit off) from the environment."""
    if psycopg is None:  # pragma: no cover
        raise EnvironmentError("psycopg is not installed")
    conninfo = _conninfo()
    try:
        return psycopg.connect(conninfo, autocommit=False)
    except psycopg.Error as exc:  # fail loud with the actual target
        raise EnvironmentError(
            f"Could not connect to PostgreSQL ({conninfo or 'PG* env'}): {exc}"
        ) from exc


def _create_temp_table(
    conn: Any,
    name: str,
    columns: Sequence[Tuple[str, str]],
    rows: Sequence[Tuple[Any, ...]],
) -> None:
    cols_ddl = ", ".join(f"{col} {sqltype}" for col, sqltype in columns)
    with conn.cursor() as cur:
        cur.execute(f"create temp table {name} ({cols_ddl}) on commit drop")
        if rows:
            placeholders = ", ".join(["%s"] * len(columns))
            cur.executemany(f"insert into {name} values ({placeholders})", list(rows))


def seed_fixture(conn: Any, fixture: str, ts_type: str) -> None:
    """Seed one named fixture (``edge`` / ``dense``) as TEMP tables.

    Creates ``p`` (parent), ``c`` (child, temporal column typed ``ts_type``) and
    ``as_of_dates``. Idempotent within a transaction: drops any prior copies
    first so a rolled-back connection can be re-seeded across ts types.
    """
    spec = preagg_cases.FIXTURES[fixture]
    with conn.cursor() as cur:
        cur.execute("drop table if exists p, c, as_of_dates")
    _create_temp_table(conn, "p", [("pid", "int")], [(k,) for k in spec["keys"]])
    _create_temp_table(
        conn,
        "c",
        [("pid", "int"), ("ts", ts_type), ("num", "numeric"), ("cat", "text")],
        spec["rows"],
    )
    _create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(spec["as_of"],)]
    )


def run_config(conn: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Render ``config`` to SQL, execute on ``conn``, return rows as dicts."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    sql = Featurizer(path, validate=False).query
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


# --- Value canonicalization (shared by capture + verification) ------------


def _canon_scalar(value: Any) -> Any:
    """Normalize one cell to a JSON-safe, comparison-stable form.

    - ``Decimal`` / numeric → ``float``
    - ``date`` / ``datetime`` → ISO string
    - ``None`` stays ``None``
    Non-finite floats raise: v0.5.2 guards every division/log, so a NaN/Inf
    here is a real regression we want surfaced loudly, not silently frozen.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float in result: {value!r}")
        return value
    # Decimal and other numerics expose __float__.
    if hasattr(value, "__float__") and not isinstance(value, str):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"non-finite numeric in result: {value!r}")
        return f
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def canonicalize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonicalize + deterministically order a result set for comparison."""
    canon = [{col: _canon_scalar(val) for col, val in row.items()} for row in rows]
    return sorted(canon, key=_order_key)


def _order_key(row: Dict[str, Any]) -> List[str]:
    """A row-ordering key immune to float summation-order noise.

    Floats are rounded before stringifying, so a value that differs only in its
    last ULP between the correlated and set-based paths (different summation
    order → e.g. ``…248523`` vs ``…24853``) sorts identically and does not
    reorder tied rows. Everything is stringified so mixed None/int/str columns
    never raise on comparison. The (unique) target id column keeps the order
    total, so golden and re-run rows align position-for-position.
    """
    key: List[str] = []
    for col in sorted(row):
        val = row[col]
        key.append(f"{round(val, 9):.9f}" if isinstance(val, float) else str(val))
    return key


def values_equal(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> Tuple[bool, str]:
    """Compare two canonical result sets. Returns ``(equal, reason)``.

    Integer/None/string cells compare exactly (including column-key set, which
    encodes the byte-identical feature names). Float cells compare with
    ``math.isclose`` so JSON round-tripping and platform FP noise don't fail a
    genuinely-equal rewrite.
    """
    if len(expected) != len(actual):
        return False, f"row count {len(expected)} != {len(actual)}"
    # Re-sort both by the noise-independent key so a float that differs only in
    # its last ULP cannot misalign otherwise-identical rows (the stored golden
    # was ordered by the correlated path's exact bits; the set-based path sums in
    # a different order). Idempotent for already-aligned sets (e.g. the gaps).
    expected = sorted(expected, key=_order_key)
    actual = sorted(actual, key=_order_key)
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e.keys() != a.keys():
            return False, (
                f"row {i}: column set differs "
                f"(expected {sorted(e)}, got {sorted(a)})"
            )
        for col in e:
            ev, av = e[col], a[col]
            if isinstance(ev, float) or isinstance(av, float):
                if ev is None or av is None:
                    if ev is not av:
                        return False, f"row {i} col {col!r}: {ev!r} != {av!r}"
                    continue
                if not math.isclose(
                    float(ev), float(av), rel_tol=rel_tol, abs_tol=abs_tol
                ):
                    return False, f"row {i} col {col!r}: {ev!r} != {av!r}"
            elif ev != av:
                return False, f"row {i} col {col!r}: {ev!r} != {av!r}"
    return True, "equal"
