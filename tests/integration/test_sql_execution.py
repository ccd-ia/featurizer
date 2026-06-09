"""Execute generated SQL against PostgreSQL and assert on real feature values.

These tests encode the *correct* expected behaviour. They skip when no database
is configured (see ``conftest.pg_conn``). The first one is a canary that simply
proves the parent->child query executes at all — the path that historically was
only ever shape-tested, never run.
"""

from __future__ import annotations

import math

import pytest

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration


# A single customer with four orders of known amounts, all before the as-of date.
_AMOUNTS = [10.0, 20.0, 30.0, 40.0]


def _seed_customer_orders(conn) -> None:
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
        [
            (1, 1, "2023-06-01", 10.0),
            (2, 1, "2023-07-01", 20.0),
            (3, 1, "2023-08-01", 30.0),
            (4, 1, "2023-09-01", 40.0),
        ],
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])


def _customer_orders_config() -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["count", "sum", "mean", "median", "min", "max", "stddev"],
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


def test_parent_child_query_executes(pg_conn):
    """Canary: the generated parent->child SQL runs and returns one row."""
    _seed_customer_orders(pg_conn)
    rows = run_featurizer(pg_conn, _customer_orders_config())
    assert len(rows) == 1


def test_parent_child_aggregation_values(pg_conn):
    """Aggregated values match the hand-computed expectations."""
    _seed_customer_orders(pg_conn)
    rows = run_featurizer(pg_conn, _customer_orders_config())
    row = rows[0]

    assert int(row["COUNT(orders.order_id)"]) == len(_AMOUNTS)
    assert float(row["SUM(orders.amount)"]) == sum(_AMOUNTS)
    assert float(row["MEAN(orders.amount)"]) == sum(_AMOUNTS) / len(_AMOUNTS)
    assert float(row["MEDIAN(orders.amount)"]) == 25.0
    assert float(row["MIN(orders.amount)"]) == min(_AMOUNTS)
    assert float(row["MAX(orders.amount)"]) == max(_AMOUNTS)
    # Sample standard deviation of [10,20,30,40] = sqrt(500/3).
    assert math.isclose(
        float(row["STDDEV(orders.amount)"]), math.sqrt(500.0 / 3.0), rel_tol=1e-9
    )


def test_interval_filter_excludes_out_of_window_events(pg_conn):
    """An interval window only aggregates events inside the daterange."""
    create_temp_table(pg_conn, "customers", [("customer_id", "int")], [(1,)])
    create_temp_table(
        pg_conn,
        "orders",
        [
            ("order_id", "int"),
            ("customer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [
            (1, 1, "2023-12-20", 100.0),  # within P1M of 2024-01-01
            (2, 1, "2023-06-01", 999.0),  # outside P1M
        ],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )

    config = _customer_orders_config()
    config["intervals"] = ["P1M"]
    rows = run_featurizer(pg_conn, config)
    row = rows[0]

    # All-time sum sees both; the P1M window sees only the recent order.
    assert float(row["SUM(orders.amount)"]) == 1099.0
    assert float(row["SUM(orders.amount|interval=P1M)"]) == 100.0
