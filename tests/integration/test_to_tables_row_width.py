"""``to_tables`` heap-row-width guard against real PostgreSQL (v1.0 hardening).

Proves the sharp edge is real — a ~1100-fixed-width-column group is a valid
*query* but its CTAS row exceeds the ~8160-byte heap page — and that the
pre-flight downshift in ``Featurizer._grouped_for_tables`` now writes several
narrower tables where the unguarded path crashed. The estimator and the
downshift decision are pinned DB-free in ``tests/test_to_tables_row_width.py``.
"""

from __future__ import annotations

import datetime
import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import HEAP_ROW_BUDGET_BYTES, estimate_heap_row_width

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

_SCHEMA = "rowwidth_it"
_N_VARS = 30


def _seed(conn) -> None:
    """One customer WITH orders (its row carries ~1100 non-null float8 values —
    the row that overflows the page) and one without (all-NULL, always fits).

    The orders sit INSIDE the shortest interval window (days before the as-of
    date): every P1W…P3Y window then contains data, so every aggregate column
    of customer 1 is non-null — NULL columns cost only a bitmap bit and would
    let the row fit, hiding the edge."""
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,), (2,)])
    var_cols = [(f"v{i}", "double precision") for i in range(_N_VARS)]
    create_temp_table(
        conn,
        "orders",
        [("order_id", "int"), ("customer_id", "int"), ("ordered_at", "date")]
        + var_cols,
        [
            (oid, 1, datetime.date(2023, 12, 26 + (oid % 4)))
            + tuple(float(oid + i) for i in range(_N_VARS))
            for oid in range(1, 5)
        ],
    )
    create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2024, 1, 1),)]
    )


def _featurizer() -> Featurizer:
    """~1100 feature columns in one lineage bucket: a valid single query whose
    unguarded CTAS row is ~9 KB of fixed-width values (float8 in, float8/bigint
    out for sum/mean/min/max/count — nothing TOASTable to squeeze)."""
    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1W", "P2W", "P1M", "P3M", "P6M", "P1Y", "P2Y", "P3Y"],
        "aggregations": ["count", "sum", "mean", "min", "max"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {f"v{i}": {"type": "numeric"} for i in range(_N_VARS)},
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


def test_unguarded_ctas_hits_row_is_too_big(pg_conn):
    """The sharp edge exists: CTAS of the (valid) single query fails on the
    heap-page limit. SELECTing the same query succeeds — the bound is
    storage-only."""
    import psycopg

    _seed(pg_conn)
    f = _featurizer()

    # Precondition: one default group, over the heap budget by estimate.
    groups = f._sharder().column_groups()
    assert len(groups) == 1
    n_cols = len(groups["group_000"])
    assert estimate_heap_row_width(n_cols + 2) > HEAP_ROW_BUDGET_BYTES

    sql = f.query  # fits_single_group: the plain single query renders
    with pg_conn.cursor() as cur:
        cur.execute("savepoint before_ctas")
        with pytest.raises(psycopg.errors.ProgramLimitExceeded) as excinfo:
            cur.execute(f'create table "unguarded_wide" as\n{sql}')
        assert "row is too big" in str(excinfo.value)
        cur.execute("rollback to savepoint before_ctas")
        # The same query SELECTs fine on this connection.
        cur.execute(sql)
        assert len(cur.fetchall()) == 2


def test_to_tables_downshifts_and_succeeds(pg_conn):
    """Where the unguarded CTAS crashed, ``to_tables`` now writes several
    heap-safe tables that re-join into the full matrix."""
    _seed(pg_conn)
    f = _featurizer()
    tables = f.to_tables(_SCHEMA, connection=pg_conn)

    assert len(tables) > 1, "expected the pre-flight to split the single group"

    seen_features: set[str] = set()
    joined: dict = {}
    with pg_conn.cursor() as cur:
        for t in tables:
            cur.execute(f"select * from {t.name}")
            cols = [d.name for d in cur.description]
            n_features = len(cols) - len(t.key_columns)
            assert (
                estimate_heap_row_width(len(cols)) <= HEAP_ROW_BUDGET_BYTES
            ), f"{t.name} was persisted over the heap budget"
            features = set(cols) - set(t.key_columns)
            assert not (features & seen_features), "columns duplicated across tables"
            seen_features |= features
            assert n_features == len(features)
            for row in cur.fetchall():
                rec = dict(zip(cols, row))
                joined.setdefault((rec["as_of_date"], rec["customer_id"]), {}).update(
                    rec
                )

    # Full coverage: the union of table columns is the whole manifest.
    assert seen_features == {e.column for e in f.feature_manifest}
    # Both customers present; the with-orders row carries non-null values.
    assert len(joined) == 2
    with_orders = next(v for (_, cid), v in joined.items() if cid == 1)
    assert any(v is not None for k, v in with_orders.items() if k.startswith("SUM("))


def test_manifest_feature_groups_match_downshifted_tables(pg_conn):
    """The persisted manifest tags every column with a really-written table."""
    _seed(pg_conn)
    f = _featurizer()
    tables = f.to_tables(_SCHEMA, connection=pg_conn)
    real_groups = {t.group for t in tables}

    with pg_conn.cursor() as cur:
        cur.execute(
            f'select "column_name", "feature_group" '
            f'from "{_SCHEMA}"."customers_manifest"'
        )
        rows = cur.fetchall()

    assert rows
    assert {g for _, g in rows} == real_groups
    assert len({c for c, _ in rows}) == len(rows)  # one row per column
