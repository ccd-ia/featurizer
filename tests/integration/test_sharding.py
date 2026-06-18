"""Execute column-group sharding against real PostgreSQL (issue #7).

Skips when no database is configured (see ``conftest.pg_conn``) or when pyarrow
(the ``[parquet]`` extra) is missing. Temp tables are created on the same
connection the queries run on, inside the rolled-back transaction.

Two things are proven here that the DB-free shape tests cannot:

* **Rejoin equivalence.** On a config small enough to *also* render unsharded,
  re-joining the column groups on ``(as_of_date, id)`` reproduces the single
  query's result exactly — same columns, same values, same NULLs.
* **The wide case that fails today.** A config whose ``<target>_transform`` CTE
  exceeds PostgreSQL's 1664-entry limit is rejected as a single query
  (``target lists can have at most 1664 entries``) yet renders as N>1 groups
  that each execute, every group under the table-column limit, with the full
  feature set reconstructed across the groups.
"""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import (
    PG_MAX_TABLE_COLUMNS,
    PG_MAX_TARGET_LIST,
    ColumnGroupSharder,
)
from featurizer.sql import SQLRenderer

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.integration

from ._harness import create_temp_table  # noqa: E402

# ------------------------------------------------------------------ #
# Fixtures / helpers
# ------------------------------------------------------------------ #


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path, validate=False)


def _seed(conn, n_vars: int) -> None:
    """Two customers (one with orders, one without) and an as-of date."""
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


def _run(conn, sql: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _narrow_config() -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1M"],
        "aggregations": ["count", "sum", "mean", "min", "max"],
        "transformations": ["identity", "abs"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"v0": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _wide_light_config() -> dict:
    """Wide enough to exceed 1664, but built from cheap primitives so each group
    *executes* on the shared test DB (no heavy ordered-set / window self-joins)."""
    variables = {f"v{i}": {"type": "numeric"} for i in range(12)}
    return {
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
                "variables": variables,
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _rejoin(group_rows: dict[str, list[dict]]) -> dict[tuple, dict]:
    """Full-outer-join group result rows on (as_of_date, customer_id)."""
    joined: dict[tuple, dict] = {}
    for rows in group_rows.values():
        for row in rows:
            key = (row["as_of_date"], row["customer_id"])
            rec = joined.setdefault(key, {})
            rec.update(row)
    return joined


# ------------------------------------------------------------------ #
# Rejoin equivalence (small config: also renders unsharded)
# ------------------------------------------------------------------ #


def test_groups_rejoin_reproduces_single_query(pg_conn):
    """Force >1 groups on a small config; the rejoin equals the single query."""
    _seed(pg_conn, n_vars=1)
    f = _featurizer(_narrow_config())

    single = _run(pg_conn, f.query)
    single_by_id = {r["customer_id"]: r for r in single}

    # Force a multi-group split with a tiny per-group budget.
    sharder = ColumnGroupSharder(f._plan, max_columns_per_group=4)
    built = sharder.build()
    assert len(built.queries) > 1, "expected the tiny budget to force >1 group"

    group_rows = {gid: _run(pg_conn, sql) for gid, sql in built.queries.items()}
    joined = _rejoin(group_rows)

    single_feature_cols = set(single[0]) - {"as_of_date", "customer_id"}

    # Same set of (as_of_date, customer_id) keys.
    single_keys = {(r["as_of_date"], r["customer_id"]) for r in single}
    assert set(joined) == single_keys

    # Same columns and same values, NULLs included.
    for cid, srow in single_by_id.items():
        jrow = next(v for (a, c), v in joined.items() if c == cid)
        rejoined_feature_cols = set(jrow) - {"as_of_date", "customer_id"}
        assert rejoined_feature_cols == single_feature_cols
        for col in single_feature_cols:
            assert srow[col] == jrow[col], f"{col} differs for customer {cid}"


def test_single_group_query_equals_query(pg_conn):
    """A config that fits returns the single query unchanged from query_groups."""
    _seed(pg_conn, n_vars=1)
    f = _featurizer(_narrow_config())
    groups = f.query_groups
    assert list(groups) == ["group_000"]
    assert groups["group_000"] == f.query
    # And it executes.
    assert len(_run(pg_conn, groups["group_000"])) == 2


def _depth3_config() -> dict:
    """stores <- orders <- items (depth 3); only orders->stores aggs are pruned."""
    return {
        "target": "stores",
        "max_depth": 3,
        "intervals": [],
        "aggregations": ["count", "sum", "mean"],
        "transformations": ["identity", "abs"],
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
                    "price": {"type": "numeric"},
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


def test_depth3_groups_rejoin_reproduces_single_query(pg_conn):
    """Depth-3 chain: forced multi-group rejoin equals the single query exactly.

    Regression guard for the deeper-chain agg CTE (``items_aggs_for_orders``),
    which must be emitted whole — pruning it against the *target's* synth columns
    would empty its select list and corrupt the orders aggregates.
    """
    f = _featurizer(_depth3_config())
    create_temp_table(pg_conn, "stores", [("store_id", "int")], [(1,), (2,)])
    create_temp_table(
        pg_conn,
        "orders",
        [
            ("order_id", "int"),
            ("store_id", "int"),
            ("ordered_at", "date"),
            ("total", "numeric"),
        ],
        [
            (10, 1, datetime.date(2023, 5, 1), 100.0),
            (11, 1, datetime.date(2023, 5, 2), 50.0),
        ],
    )
    create_temp_table(
        pg_conn,
        "items",
        [
            ("item_id", "int"),
            ("order_id", "int"),
            ("added_at", "date"),
            ("price", "numeric"),
        ],
        [
            (100, 10, datetime.date(2023, 5, 1), 20.0),
            (101, 10, datetime.date(2023, 5, 1), 30.0),
            (102, 11, datetime.date(2023, 5, 2), 5.0),
        ],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2023, 7, 1),)]
    )

    single = _run(pg_conn, f.query)
    single_by_id = {r["store_id"]: r for r in single}

    built = ColumnGroupSharder(f._plan, max_columns_per_group=3).build()
    assert len(built.queries) > 1

    group_rows = {gid: _run(pg_conn, sql) for gid, sql in built.queries.items()}
    joined: dict = {}
    for rows in group_rows.values():
        for row in rows:
            joined.setdefault((row["as_of_date"], row["store_id"]), {}).update(row)

    single_feature_cols = set(single[0]) - {"as_of_date", "store_id"}
    for sid, srow in single_by_id.items():
        jrow = next(v for (a, s), v in joined.items() if s == sid)
        assert set(jrow) - {"as_of_date", "store_id"} == single_feature_cols
        for col in single_feature_cols:
            assert srow[col] == jrow[col], f"{col} differs for store {sid}"


# ------------------------------------------------------------------ #
# The wide case that fails today
# ------------------------------------------------------------------ #


def test_wide_single_query_rejected_by_postgres(pg_conn):
    """The unsharded wide query trips the 1664-entry target-list limit."""
    import psycopg

    _seed(pg_conn, n_vars=12)
    f = _featurizer(_wide_light_config())
    # Bypass the Python-side guard to prove PostgreSQL itself rejects it.
    raw_single = SQLRenderer().render(f._plan)
    # TooManyColumns is the specific subclass of the program-limit error
    # PostgreSQL raises for an over-wide target list.
    with pytest.raises(psycopg.errors.TooManyColumns) as excinfo:
        _run(pg_conn, raw_single)
    assert "1664" in str(excinfo.value)


def test_wide_query_property_raises(pg_conn):
    """`.query` refuses the wide config and points at the sharded API."""
    _seed(pg_conn, n_vars=12)
    f = _featurizer(_wide_light_config())
    with pytest.raises(ValueError, match="too wide"):
        _ = f.query


def test_wide_groups_each_execute_and_cover_all_features(pg_conn):
    """N>1 groups each execute, each under the limit, full feature coverage."""
    _seed(pg_conn, n_vars=12)
    f = _featurizer(_wide_light_config())

    groups = f.query_groups
    assert len(groups) > 1

    full_features = {c.name for c in f._plan.cte_specs["customers_transform"].columns}

    group_rows: dict[str, list[dict]] = {}
    seen_features: set[str] = set()
    for gid, sql in groups.items():
        rows = _run(pg_conn, sql)
        group_rows[gid] = rows
        out_cols = set(rows[0]) if rows else set()
        assert (
            len(out_cols) <= PG_MAX_TABLE_COLUMNS
        ), f"{gid} returned {len(out_cols)} columns, over the table limit"
        assert len(out_cols) <= PG_MAX_TARGET_LIST
        # PostgreSQL truncates identifiers to 63 bytes, so compare on the
        # truncated forms the planner would also produce.
        feats = out_cols - {"as_of_date", "customer_id"}
        overlap = seen_features & feats
        assert not overlap, f"feature columns duplicated across groups: {overlap}"
        seen_features |= feats

    # Every group carries the two join keys.
    for rows in group_rows.values():
        if rows:
            assert "as_of_date" in rows[0] and "customer_id" in rows[0]

    # Coverage: the union of group features equals the full set (compared on the
    # 63-byte-truncated names PostgreSQL returns).
    def trunc(name: str) -> str:
        bare = name[1:-1] if name.startswith('"') else name
        return bare[:63]

    expected = {trunc(n) for n in full_features}
    assert (
        seen_features == expected
    ), f"missing: {expected - seen_features}; extra: {seen_features - expected}"


# ------------------------------------------------------------------ #
# Arrow / Parquet grouped output
# ------------------------------------------------------------------ #


def test_to_arrow_returns_group_dict_for_wide_config(pg_conn):
    """to_arrow yields an OrderedDict of joinable pyarrow tables when wide."""
    from collections import OrderedDict

    _seed(pg_conn, n_vars=12)
    f = _featurizer(_wide_light_config())
    out = f.to_arrow(connection=pg_conn)

    assert isinstance(out, OrderedDict)
    assert len(out) > 1
    for gid, table in out.items():
        assert isinstance(table, pa.Table)
        assert table.column_names[:2] == ["as_of_date", "customer_id"]


def test_to_arrow_returns_single_table_for_narrow_config(pg_conn):
    """to_arrow preserves the single-Table contract for a config that fits."""
    _seed(pg_conn, n_vars=1)
    f = _featurizer(_narrow_config())
    out = f.to_arrow(connection=pg_conn)
    assert isinstance(out, pa.Table)
    assert out.column_names[:2] == ["as_of_date", "customer_id"]


def test_to_parquet_writes_one_file_per_group(pg_conn, tmp_path: Path):
    """to_parquet writes a directory of group_<NNN>.parquet for a wide config."""
    _seed(pg_conn, n_vars=12)
    f = _featurizer(_wide_light_config())

    out_dir = tmp_path / "matrix"
    f.to_parquet(str(out_dir), connection=pg_conn)

    files = sorted(out_dir.glob("group_*.parquet"))
    assert len(files) > 1

    # Each file re-joins on (as_of_date, customer_id) and reconstructs the matrix.
    all_features: set[str] = set()
    n_rows = None
    for path in files:
        table = pq.read_table(path)
        assert table.column_names[:2] == ["as_of_date", "customer_id"]
        if n_rows is None:
            n_rows = table.num_rows
        assert table.num_rows == n_rows, "every group must have the same row count"
        all_features |= set(table.column_names) - {"as_of_date", "customer_id"}

    full = {
        c.name[:63].strip('"') for c in f._plan.cte_specs["customers_transform"].columns
    }
    assert all_features == full


def test_to_parquet_writes_single_file_for_narrow_config(pg_conn, tmp_path: Path):
    """to_parquet writes one file at the given path for a config that fits."""
    _seed(pg_conn, n_vars=1)
    f = _featurizer(_narrow_config())
    out = tmp_path / "features.parquet"
    f.to_parquet(str(out), connection=pg_conn)
    assert out.exists() and out.is_file()
    assert pq.read_table(out).num_rows == 2


# ------------------------------------------------------------------ #
# Temp-table materialization of an oversized non-target child chain (issue #7).
# Forced on a small depth-3 config via ``materialize_threshold=1`` so the whole
# child chain (items_aggs_for_orders -> orders_synth -> orders_transform) is
# materialized into TEMP shards, without needing a genuinely 1664-wide CTE.
# ------------------------------------------------------------------ #


def _seed_depth3(conn) -> None:
    """stores <- orders <- items, with one store that has orders/items and one
    that has none (so the LEFT JOINs must preserve NULLs)."""
    create_temp_table(conn, "stores", [("store_id", "int")], [(1,), (2,)])
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("store_id", "int"),
            ("ordered_at", "date"),
            ("total", "numeric"),
        ],
        [
            (10, 1, datetime.date(2023, 5, 1), 100.0),
            (11, 1, datetime.date(2023, 5, 2), 50.0),
        ],
    )
    create_temp_table(
        conn,
        "items",
        [
            ("item_id", "int"),
            ("order_id", "int"),
            ("added_at", "date"),
            ("price", "numeric"),
        ],
        [
            (100, 10, datetime.date(2023, 5, 1), 20.0),
            (101, 10, datetime.date(2023, 5, 1), 30.0),
            (102, 11, datetime.date(2023, 5, 2), 5.0),
        ],
    )
    create_temp_table(
        conn, "as_of_dates", [("as_of_date", "date")], [(datetime.date(2023, 7, 1),)]
    )


def _materialized_featurizer(config: dict):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path, validate=False, materialize_threshold=1)


def test_materialized_chain_rejoin_equals_single_query(pg_conn):
    """Forcing the child chain into TEMP-table shards, then running the preamble +
    group queries on one connection and re-joining, reproduces the single
    (non-materialized) query's matrix exactly — same keys, columns, values, NULLs."""
    _seed_depth3(pg_conn)

    # Baseline: the ordinary single query (no materialization).
    single = _run(pg_conn, _featurizer(_depth3_config()).query)
    single_by_id = {r["store_id"]: r for r in single}
    feature_cols = set(single[0]) - {"as_of_date", "store_id"}

    # Materialized: tiny threshold pushes the whole non-target chain to temp tables.
    f = _materialized_featurizer(_depth3_config())
    grouped = f._grouped()
    assert grouped.materialization is not None, "expected a materialization preamble"
    assert grouped.materialization.ddl

    with pg_conn.cursor() as cur:
        for ddl in grouped.materialization.ddl:
            cur.execute(ddl)

    joined: dict = {}
    for sql in grouped.queries.values():
        for row in _run(pg_conn, sql):
            joined.setdefault((row["as_of_date"], row["store_id"]), {}).update(row)

    # Same (as_of_date, store_id) keys.
    assert set(joined) == {(r["as_of_date"], r["store_id"]) for r in single}
    # Same columns and values, NULLs included.
    for sid, srow in single_by_id.items():
        jrow = next(v for (a, s), v in joined.items() if s == sid)
        assert set(jrow) - {"as_of_date", "store_id"} == feature_cols
        for col in feature_cols:
            assert srow[col] == jrow[col], f"{col} differs for store {sid}"


def test_to_arrow_runs_materialization_preamble(pg_conn):
    """``to_arrow`` runs the TEMP-table preamble on its connection, so a config
    whose child chain must be materialized executes end-to-end and the values
    match the single-query baseline."""
    _seed_depth3(pg_conn)
    single = _run(pg_conn, _featurizer(_depth3_config()).query)
    single_by_id = {r["store_id"]: r for r in single}

    f = _materialized_featurizer(_depth3_config())
    out = f.to_arrow(connection=pg_conn)

    # The small target fits one group, so a single table comes back.
    table = out if not isinstance(out, dict) else next(iter(out.values()))
    rows = table.to_pylist()
    by_id = {r["store_id"]: r for r in rows}
    assert set(by_id) == set(single_by_id)
    for sid, srow in single_by_id.items():
        arow = by_id[sid]
        for col in set(single[0]) - {"as_of_date", "store_id"}:
            assert arow[col] == srow[col], f"{col} differs for store {sid}"
