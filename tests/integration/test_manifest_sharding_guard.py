"""Manifest-under-sharding guard (v1.0 hardening).

The persisted ``<stem>_manifest`` is lineage triage consumes: every row's
``feature_group`` must name a really-persisted ``<stem>_group_<NNN>`` table,
and every output column must land in exactly one group table. The silent
``group_000`` fallback that could mis-tag a manifest row is gone — an
orphaned column now raises before anything is inserted (pinned DB-free in
``tests/test_to_tables_row_width.py``); this file proves the invariants hold
end-to-end on a genuinely sharded config against real PostgreSQL.
"""

from __future__ import annotations

import datetime
import tempfile

import pytest
import yaml

from featurizer import Featurizer

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

_SCHEMA = "manifest_shard_it"


def _seed(conn, n_vars: int = 12) -> None:
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,), (2,)])
    var_cols = [(f"v{i}", "numeric") for i in range(n_vars)]
    create_temp_table(
        conn,
        "orders",
        [("order_id", "int"), ("customer_id", "int"), ("ordered_at", "date")]
        + var_cols,
        [
            (oid, 1, datetime.date(2023, 6, (oid % 28) + 1))
            + tuple(float(oid + i) for i in range(n_vars))
            for oid in range(1, 5)
        ],
    )
    create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )


def _wide_featurizer() -> Featurizer:
    """Exceeds the 1664-entry target-list limit, so ``to_tables`` genuinely
    shards into several group tables (cheap primitives only, so every group
    executes on the shared test database)."""
    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1W", "P1M", "P3M", "P6M", "P1Y", "P2Y"],
        "aggregations": ["count", "sum", "mean", "min", "max"],
        "transformations": ["identity", "abs", "sqrt"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {f"v{i}": {"type": "numeric"} for i in range(12)},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return Featurizer(handle.name, validate=False)


def test_sharded_manifest_invariants(pg_conn):
    """Every manifest row maps to a persisted table; every output column
    appears exactly once across the group tables; the mapping round-trips."""
    _seed(pg_conn)
    f = _wide_featurizer()
    tables = f.to_tables(_SCHEMA, connection=pg_conn)
    assert len(tables) > 1, "expected a genuinely sharded config"

    # What was actually persisted, per group table.
    persisted: dict[str, set[str]] = {}
    with pg_conn.cursor() as cur:
        for t in tables:
            cur.execute(f"select * from {t.name} limit 0")
            persisted[t.group] = {d.name for d in cur.description} - set(t.key_columns)
        cur.execute(
            f'select "column_name", "feature_group" '
            f'from "{_SCHEMA}"."customers_manifest"'
        )
        manifest_rows = cur.fetchall()

    # 1. Every row's feature_group is a real persisted table.
    assert {g for _, g in manifest_rows} <= set(persisted)

    # 2. Every output column appears exactly once across the group tables.
    all_columns = [c for cols in persisted.values() for c in cols]
    assert len(all_columns) == len(set(all_columns))

    # 3. The manifest tags each column with the table that actually holds it,
    #    covering the full output — no orphans in either direction.
    assert {c for c, _ in manifest_rows} == set(all_columns)
    for column_name, feature_group in manifest_rows:
        assert (
            column_name in persisted[feature_group]
        ), f"{column_name} tagged {feature_group} but persisted elsewhere"
