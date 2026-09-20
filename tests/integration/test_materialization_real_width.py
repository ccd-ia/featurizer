"""The TEMP-table path at the width it exists for (issue #37).

Every other materialization test forces the path with ``materialize_threshold=1``
on a narrow config. That proves the shape of the preamble and nothing about the
limit it is there to get under: a materialized CTE is by definition wider than
PostgreSQL's 1664-entry target list, and the path re-joined its shards with
``select *`` — itself over the limit, in a FROM subquery, a CTE or a lateral
alike. Measured on de03142: the first statement that READ a materialized CTE
failed, ``target lists can have at most 1664 entries``.

Here the threshold is the default and the child chain is genuinely too wide,
the shape issue #37 measured: 150 numeric variables x 5 aggregations x
(2 intervals + the whole history) = 1,813 columns. The data is non-null
``float8`` in every column, because a row of NULLs takes no room and would hide
the second limit, the ~8 kB heap row (issue #52).

The single query cannot run at this width, which is the point, so the oracle is
a NARROW config over the same tables: three of the 150 variables, one query.
Every column the two results share has to be equal.
"""

from __future__ import annotations

import datetime
import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import PG_MAX_TARGET_LIST

pytestmark = pytest.mark.integration

N_VARIABLES = 150
NARROW = (0, 75, 149)
AS_OF = [datetime.date(2023, 7, 1), datetime.date(2023, 8, 1)]


def _variable(i: int) -> str:
    return f"v{i:03d}"


def _config(variables: list[str]) -> dict:
    return {
        "target": "stores",
        "max_depth": 3,
        "intervals": ["P30D", "P90D"],
        "aggregations": ["count", "sum", "mean", "min", "max"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "stores", "table": "stores", "id": "store_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {
                    "store_id": {"type": "index"},
                    "total": {"type": "numeric"},
                },
            },
            {
                "alias": "items",
                "table": "items",
                "id": "item_id",
                "temporal_ix": "added_at",
                "variables": {
                    "order_id": {"type": "index"},
                    **{name: {"type": "numeric"} for name in variables},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "stores", "key": "store_id"},
                "child": {"entity": "orders", "key": "store_id"},
            },
            {
                "parent": {"entity": "orders", "key": "order_id"},
                "child": {"entity": "items", "key": "order_id"},
            },
        ],
    }


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
    return Featurizer(handle.name, validate=False)


def _seed(conn) -> None:
    columns = ", ".join(f"{_variable(i)} double precision" for i in range(N_VARIABLES))
    with conn.cursor() as cur:
        cur.execute("create temp table stores (store_id int) on commit drop")
        cur.execute("insert into stores values (1), (2)")
        cur.execute(
            "create temp table orders (order_id int, store_id int, ordered_at date, "
            "total double precision) on commit drop"
        )
        cur.execute(
            "insert into orders values (10, 1, '2023-05-01', 100), "
            "(11, 1, '2023-06-20', 50), (12, 1, '2023-07-15', 7), "
            "(13, 1, '2023-09-30', 1), (14, 2, '2023-06-01', 9)"
        )
        cur.execute(
            "create temp table items (item_id int, order_id int, added_at date, "
            f"{columns}) on commit drop"
        )
        # Every variable non-null and different per row: v_i = item_id + i / 100.
        values = ", ".join(f"g.item_id + {i} / 100.0" for i in range(N_VARIABLES))
        cur.execute(
            "insert into items select g.item_id, g.order_id, g.added_at, "
            f"{values} from (values (100, 10, date '2023-05-01'), "
            "(101, 10, date '2023-05-02'), (102, 11, date '2023-06-20'), "
            "(103, 12, date '2023-07-15'), (104, 13, date '2023-09-30'), "
            "(105, 14, date '2023-06-01')) g(item_id, order_id, added_at)"
        )
        cur.execute("create temp table as_of_dates (as_of_date date) on commit drop")
        cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in AS_OF])


def _rows(conn, sql: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql)
        names = [d.name for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def test_a_child_chain_wider_than_the_target_list_limit_runs_and_is_right(pg_conn):
    wide = _featurizer(_config([_variable(i) for i in range(N_VARIABLES)]))
    widths = {
        name: len(spec.key_columns) + len(spec.columns)
        for name, spec in wide._plan.cte_specs.items()
    }
    assert widths["orders_transform"] > PG_MAX_TARGET_LIST, widths

    _seed(pg_conn)
    with pg_conn.cursor() as cur:
        # A group query here carries ~1,400 aggregate expressions, and
        # PostgreSQL's JIT compiles every one of them before reading the five
        # rows: measured 55.6 s per group with it on, 0.1 s with it off. This
        # test is about two limits, not about that.
        cur.execute("set local jit = off")
    narrow = _featurizer(_config([_variable(i) for i in NARROW]))
    expected = {
        (row["as_of_date"], row["store_id"]): row
        for row in _rows(pg_conn, narrow.query)
    }

    grouped = wide._grouped()
    assert grouped.materialization is not None, "expected a TEMP-table preamble"
    with pg_conn.cursor() as cur:
        for statement in grouped.materialization.ddl:
            cur.execute(statement)
    actual: dict = {}
    for sql in grouped.queries.values():
        for row in _rows(pg_conn, sql):
            actual.setdefault((row["as_of_date"], row["store_id"]), {}).update(row)

    assert set(actual) == set(expected)
    shared = set(next(iter(expected.values()))) & set(next(iter(actual.values())))
    # Both configs name these the same way, so the narrow one's columns are a
    # subset of the wide one's (3 variables of the 150).
    assert shared == set(next(iter(expected.values())))
    assert len(shared) > 100
    differing = [
        (key, column, expected[key][column], actual[key][column])
        for key in expected
        for column in shared
        if expected[key][column] != actual[key][column]
    ]
    assert not differing, f"{len(differing)} cells differ, e.g. {differing[:3]}"
    # Not all NULL, or equality would prove nothing.
    assert any(
        value is not None
        for row in expected.values()
        for column, value in row.items()
        if column not in ("as_of_date", "store_id")
    )
