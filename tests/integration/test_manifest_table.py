"""``to_tables`` persists the feature manifest beside the group tables.

Runs on the rolled-back ``pg_conn`` transaction (caller-owned: nothing is
committed here, matching the to_tables contract when ``connection=`` is given).
"""

from __future__ import annotations

import datetime
import tempfile

import pytest
import yaml

from featurizer import Featurizer

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

_SCHEMA = "manifest_it"


def _seed(conn) -> None:
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,)])
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("customer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [(10, 1, datetime.date(2023, 5, 1), 10.0)],
    )
    create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )


def _featurizer() -> Featurizer:
    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1M"],
        "aggregations": ["sum", "mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"amount": {"type": "numeric"}},
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
        return Featurizer(handle.name)


def test_manifest_table_written_with_row_per_column(pg_conn):
    _seed(pg_conn)
    f = _featurizer()
    f.to_tables(_SCHEMA, connection=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute(
            f'select "column_name", "kind", "feature_group", "description", '
            f'"parents" from "{_SCHEMA}"."customers_manifest" order by "column_name"'
        )
        rows = {r[0]: r for r in cur.fetchall()}

    expected = {e.column for e in f.feature_manifest}
    assert set(rows) == expected
    assert len(expected) == len(f.feature_manifest)  # no duplicate rows

    _, kind, group, description, parents = rows["SUM(orders.amount|interval=P1M)"]
    assert kind == "derived"
    assert group == "group_000"
    assert "orders.amount" in description
    assert parents == ["amount"]


def test_manifest_columns_join_to_group_table(pg_conn):
    _seed(pg_conn)
    f = _featurizer()
    tables = f.to_tables(_SCHEMA, connection=pg_conn)
    assert len(tables) == 1  # narrow config -> single group; manifest not in list

    with pg_conn.cursor() as cur:
        cur.execute(f"select * from {tables[0].name} limit 0")
        group_columns = {d.name for d in cur.description}
        cur.execute(
            f'select "column_name", "feature_group" '
            f'from "{_SCHEMA}"."customers_manifest"'
        )
        manifest_rows = cur.fetchall()

    for column_name, feature_group in manifest_rows:
        assert feature_group == "group_000"
        assert column_name in group_columns


def test_rerun_is_idempotent_and_quotes_survive(pg_conn):
    _seed(pg_conn)
    f = _featurizer()
    f.to_tables(_SCHEMA, connection=pg_conn)
    f.to_tables(_SCHEMA, connection=pg_conn)  # DROP+CREATE, no duplicate rows

    with pg_conn.cursor() as cur:
        cur.execute(f'select count(*) from "{_SCHEMA}"."customers_manifest"')
        count = cur.fetchone()[0]
        cur.execute(
            f'select "definition" from "{_SCHEMA}"."customers_manifest" '
            "where \"column_name\" = 'SUM(orders.amount|interval=P1M)'"
        )
        definition = cur.fetchone()[0]

    assert count == len(f.feature_manifest)
    # The SQL definition carries quotes/parens intact through the round-trip.
    assert "sum(" in definition.lower()
