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


def test_asof_direct_join_executes(pg_conn):
    """Point-in-time (as-of) direct join pulls the most recent in-grace value."""
    create_temp_table(
        pg_conn,
        "patients",
        [("patient_id", "int"), ("registered_at", "date"), ("age", "numeric")],
        [(1, "2023-06-01", 40.0)],
    )
    create_temp_table(
        pg_conn,
        "care_plans",
        [
            ("plan_id", "int"),
            ("patient_id", "int"),
            ("effective_at", "date"),
            ("risk_score", "numeric"),
        ],
        [(1, 1, "2023-05-20", 0.8)],  # within P14D grace of registered_at
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )

    config = {
        "target": "patients",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["mean"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "patients",
                "table": "patients",
                "id": "patient_id",
                "temporal_ix": "registered_at",
                "variables": {"age": {"type": "numeric"}},
            },
            {
                "alias": "care_plans",
                "table": "care_plans",
                "id": "plan_id",
                "temporal_ix": "effective_at",
                "variables": {
                    "patient_id": {"type": "index"},
                    "risk_score": {"type": "numeric"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "care_plans", "key": "patient_id"},
                "child": {"entity": "patients", "key": "patient_id"},
                "temporal": {"mode": "as_of", "grace": "P14D"},
            }
        ],
    }
    rows = run_featurizer(pg_conn, config)
    assert len(rows) == 1
    row = rows[0]
    # The as-of join pulls the care plan effective within the P14D grace window
    # (2023-05-20 is 12 days before registration on 2023-06-01).
    risk_cols = [k for k in row if "risk_score" in k]
    assert risk_cols, f"no risk_score column pulled; got {list(row)}"
    assert all(float(row[k]) == 0.8 for k in risk_cols)


def test_text_lexical_features_compute_correct_values(pg_conn):
    """Text Path-1 lexical transformers produce correct numeric scores."""
    create_temp_table(pg_conn, "authors", [("author_id", "int")], [(1,)])
    create_temp_table(
        pg_conn,
        "posts",
        [
            ("post_id", "int"),
            ("author_id", "int"),
            ("created_at", "date"),
            ("body", "text"),
        ],
        [(1, 1, "2023-01-01", "Hello world! This is GREAT")],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )

    config = {
        "target": "authors",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["mean", "max"],
        "transformations": [
            "num_chars",
            "num_words",
            "caps_ratio",
            "exclamation_count",
            "unique_word_ratio",
        ],
        "entities": [
            {"alias": "authors", "table": "authors", "id": "author_id"},
            {
                "alias": "posts",
                "table": "posts",
                "id": "post_id",
                "temporal_ix": "created_at",
                "variables": {"body": {"type": "text"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "authors", "key": "author_id"},
                "child": {"entity": "posts", "key": "author_id"},
            }
        ],
    }
    rows = run_featurizer(pg_conn, config)
    assert len(rows) == 1
    row = rows[0]

    def value(substring: str) -> float:
        matches = [k for k in row if substring in k]
        assert matches, f"no column matching {substring!r}; got {list(row)}"
        return float(row[matches[0]])

    # "Hello world! This is GREAT": 26 chars, 5 words, 1 '!', all words distinct,
    # 7 uppercase of 21 letters.
    assert value("MEAN(posts.NUM_CHARS") == 26.0
    assert value("MEAN(posts.NUM_WORDS") == 5.0
    assert value("MAX(posts.EXCLAMATION_COUNT") == 1.0
    assert value("MEAN(posts.UNIQUE_WORD_RATIO") == 1.0
    assert math.isclose(value("MEAN(posts.CAPS_RATIO"), 7.0 / 21.0, rel_tol=1e-6)


def test_graph_degree_features_with_causal_bound(pg_conn):
    """Degree features over an edge table, bounded as-of the cutoff."""
    create_temp_table(pg_conn, "users", [("user_id", "int")], [(1,), (2,), (3,)])
    create_temp_table(
        pg_conn,
        "follows",
        [("follower_id", "int"), ("followee_id", "int"), ("created_at", "date")],
        [
            (1, 2, "2023-01-01"),
            (1, 3, "2023-02-01"),
            (2, 1, "2023-03-01"),
            (3, 1, "2023-06-15"),  # after the early cutoff
        ],
    )
    create_temp_table(
        pg_conn,
        "as_of_dates",
        [("as_of_date", "date")],
        [("2023-04-01",), ("2023-12-31",)],
    )

    config = {
        "target": "users",
        "max_depth": 1,
        "intervals": [],
        "aggregations": ["mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "users", "table": "users", "id": "user_id"},
            {
                "alias": "follows",
                "table": "follows",
                "edge": {
                    "node": "users",
                    "source": "follower_id",
                    "target": "followee_id",
                    "timestamp": "created_at",
                },
            },
        ],
    }
    rows = run_featurizer(pg_conn, config)

    def degree(as_of: str, user_id: int, metric: str):
        row = next(
            r
            for r in rows
            if str(r["as_of_date"]) == as_of and r.get("user_id") == user_id
        )
        col = next(c for c in row if c.startswith(metric + "("))
        return row[col]

    # Full graph at 2023-12-31: user 1 follows 2 and 3, is followed by 2 and 3.
    assert degree("2023-12-31", 1, "OUT_DEGREE") == 2
    assert degree("2023-12-31", 1, "IN_DEGREE") == 2
    assert degree("2023-12-31", 1, "DEGREE") == 4
    # Causal bound at 2023-04-01: the 2023-06-15 edge (3->1) is excluded.
    assert degree("2023-04-01", 1, "IN_DEGREE") == 1
