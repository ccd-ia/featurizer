"""Execute relationship topologies that the fixtures never covered.

Phase 1 (bug 1): relationships whose parent/child key column names differ —
aggregation, direct-transfer, and as-of directions, plus the issue-#7
temp-table materialization path with the corrected join key.

All tests run on the rolled-back ``pg_conn`` transaction; they skip when no
PostgreSQL is configured.
"""

from __future__ import annotations

import datetime
import tempfile

import pytest
import yaml

from featurizer import Featurizer

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------ #
# Aggregation direction: customers.customer_id <- orders.buyer_id
# ------------------------------------------------------------------ #


def _seed_differing_keys(conn) -> None:
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,), (2,)])
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("buyer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [
            (10, 1, datetime.date(2023, 5, 1), 10.0),
            (11, 1, datetime.date(2023, 6, 1), 20.0),
            (12, 1, datetime.date(2023, 7, 1), 30.0),
        ],
    )
    create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )


def _differing_keys_config() -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["count", "sum", "mean"],
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
                "child": {"entity": "orders", "key": "buyer_id"},
            }
        ],
    }


def test_differing_keys_aggregation_executes_with_correct_values(pg_conn):
    _seed_differing_keys(pg_conn)
    rows = run_featurizer(pg_conn, _differing_keys_config())
    by_id = {r["customer_id"]: r for r in rows}
    assert set(by_id) == {1, 2}
    assert float(by_id[1]["SUM(orders.amount)"]) == 60.0
    assert float(by_id[1]["MEAN(orders.amount)"]) == 20.0
    # Customer 2 has no orders: LEFT JOIN must preserve the row with NULLs.
    assert by_id[2]["SUM(orders.amount)"] is None


def test_differing_keys_materialized_path_matches_single_query(pg_conn):
    """issue #7: with the corrected child_key join key, the TEMP-shard
    materialization reproduces the single-query matrix on a differing-key
    relationship."""
    _seed_differing_keys(pg_conn)
    baseline = {
        r["customer_id"]: r for r in run_featurizer(pg_conn, _differing_keys_config())
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(_differing_keys_config(), handle)
        path = handle.name
    f = Featurizer(path, validate=False, materialize_threshold=1)
    grouped = f._grouped()
    assert grouped.materialization is not None
    assert grouped.materialization.ddl

    with pg_conn.cursor() as cur:
        for ddl in grouped.materialization.ddl:
            cur.execute(ddl)
        merged: dict[int, dict] = {}
        for sql in grouped.queries.values():
            cur.execute(sql)
            columns = [d.name for d in cur.description]
            for row in cur.fetchall():
                rec = dict(zip(columns, row))
                merged.setdefault(rec["customer_id"], {}).update(rec)

    assert set(merged) == set(baseline)
    for cid, base_row in baseline.items():
        for col, expected in base_row.items():
            assert merged[cid][col] == expected, (cid, col)


# ------------------------------------------------------------------ #
# Direct-transfer direction: orders (target) pulls customers via
# customers.customer_ref = orders.buyer_id, where customer_ref != cust_pk.
# ------------------------------------------------------------------ #


def test_direct_transfer_with_non_id_parent_key(pg_conn):
    create_temp_table(
        conn := pg_conn,
        "customers",
        [("cust_pk", "int"), ("customer_ref", "int"), ("score", "numeric")],
        [(900, 1, 7.5), (901, 2, 3.0)],
    )
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("buyer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [
            (10, 1, datetime.date(2023, 5, 1), 10.0),
            (11, 2, datetime.date(2023, 6, 1), 20.0),
        ],
    )
    create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )
    config = {
        "target": "orders",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["sum"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "customers",
                "table": "customers",
                "id": "cust_pk",
                "variables": {"score": {"type": "numeric"}},
            },
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
                "parent": {"entity": "customers", "key": "customer_ref"},
                "child": {"entity": "orders", "key": "buyer_id"},
            }
        ],
    }
    rows = run_featurizer(pg_conn, config)
    by_order = {r["order_id"]: r for r in rows}
    assert float(by_order[10]["score"]) == 7.5
    assert float(by_order[11]["score"]) == 3.0


# ------------------------------------------------------------------ #
# As-of direction: patients (target) pulls the latest care plan via
# care_plans.patient_ref = patients.patient_id, patient_ref != plan_id.
# ------------------------------------------------------------------ #


def test_asof_with_non_id_parent_key(pg_conn):
    create_temp_table(
        pg_conn,
        "patients",
        [("patient_id", "int"), ("admission_date", "date"), ("age", "numeric")],
        [(1, datetime.date(2023, 6, 15), 40.0)],
    )
    create_temp_table(
        pg_conn,
        "care_plans",
        [
            ("plan_id", "int"),
            ("patient_ref", "int"),
            ("plan_date", "date"),
            ("cost", "numeric"),
        ],
        [
            # Older plan, then a newer one still before admission: the as-of
            # lateral must pick the newer (100.0), not the older (50.0).
            (500, 1, datetime.date(2023, 6, 1), 50.0),
            (501, 1, datetime.date(2023, 6, 10), 100.0),
            # A plan after admission must never match.
            (502, 1, datetime.date(2023, 7, 1), 999.0),
        ],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )
    config = {
        "target": "patients",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["sum"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "patients",
                "table": "patients",
                "id": "patient_id",
                "temporal_ix": "admission_date",
                "variables": {"age": {"type": "numeric"}},
            },
            {
                "alias": "care_plans",
                "table": "care_plans",
                "id": "plan_id",
                "temporal_ix": "plan_date",
                "variables": {"cost": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "care_plans", "key": "patient_ref"},
                "child": {"entity": "patients", "key": "patient_id"},
                "temporal": {"mode": "as_of", "grace": "P30D"},
            }
        ],
    }
    rows = run_featurizer(pg_conn, config)
    assert len(rows) == 1
    assert float(rows[0]["cost"]) == 100.0


# ------------------------------------------------------------------ #
# Phase 2: parallel relationships (buyer/seller) execute independently.
# ------------------------------------------------------------------ #


def test_parallel_relationships_aggregate_independently(pg_conn):
    """Two named relationships between one entity pair produce different
    values per role: buyer-side totals != seller-side totals."""
    create_temp_table(pg_conn, "customers", [("customer_id", "int")], [(1,), (2,)])
    create_temp_table(
        pg_conn,
        "orders",
        [
            ("order_id", "int"),
            ("buyer_id", "int"),
            ("seller_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [
            # Customer 1 buys 10+20=30 and sells 100; customer 2 buys 100
            # and sells 30. Totals are asymmetric on purpose.
            (10, 1, 2, datetime.date(2023, 5, 1), 10.0),
            (11, 1, 2, datetime.date(2023, 6, 1), 20.0),
            (12, 2, 1, datetime.date(2023, 7, 1), 100.0),
        ],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )
    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["sum", "count"],
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
                "name": "purchases",
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "buyer_id"},
            },
            {
                "name": "sales",
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "seller_id"},
            },
        ],
    }
    rows = run_featurizer(pg_conn, config)
    by_id = {r["customer_id"]: r for r in rows}
    assert float(by_id[1]["SUM(purchases.amount)"]) == 30.0
    assert float(by_id[1]["SUM(sales.amount)"]) == 100.0
    assert float(by_id[2]["SUM(purchases.amount)"]) == 100.0
    assert float(by_id[2]["SUM(sales.amount)"]) == 30.0
