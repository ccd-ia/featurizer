"""Execute the COPY -> Arrow / Parquet export path against real PostgreSQL.

These tests skip when no database is configured (see ``conftest.pg_conn``) or
when pyarrow (the ``[parquet]`` extra) is not installed. They run ``COPY`` on the
same connection that holds the session ``TEMP`` tables, so the rendered query's
table references resolve.
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from featurizer import Featurizer

from ._harness import create_temp_table, run_featurizer

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.integration


def _seed_two_customers(conn) -> None:
    """Customer 1 has four orders before the as-of date; customer 2 has none."""
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,), (2,)])
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


def _config() -> dict:
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


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path)


def test_to_arrow_schema_and_nulls(pg_conn):
    """to_arrow returns a Table with keys as columns and SQL NULLs as Arrow nulls."""
    _seed_two_customers(pg_conn)
    table = _featurizer(_config()).to_arrow(connection=pg_conn)

    assert isinstance(table, pa.Table)
    # keys are ordinary leading columns (not an index)
    assert table.column_names[:2] == ["as_of_date", "customer_id"]

    d = table.to_pydict()
    row2 = d["customer_id"].index(2)
    # customer 2 has no orders -> every aggregate is an Arrow null (not NaN)
    for name in table.column_names:
        if name in ("as_of_date", "customer_id"):
            continue
        assert d[name][row2] is None, f"{name} should be null for the no-orders row"

    # numeric aggregates land as float64 (numeric_as_float default), not decimal
    assert pa.types.is_floating(table.column("MEAN(orders.amount)").type)


def test_to_arrow_matches_direct_fetch(pg_conn):
    """Arrow values equal a plain cursor fetch of the same query (no pandas hop)."""
    _seed_two_customers(pg_conn)
    config = _config()
    rows = run_featurizer(pg_conn, config)  # plain dicts via cursor.fetchall
    table = _featurizer(config).to_arrow(connection=pg_conn)

    arrow_rows = table.to_pylist()
    assert len(arrow_rows) == len(rows)

    by_id_fetch = {r["customer_id"]: r for r in rows}
    by_id_arrow = {r["customer_id"]: r for r in arrow_rows}
    assert by_id_fetch.keys() == by_id_arrow.keys()

    for cust_id, fetched in by_id_fetch.items():
        got = by_id_arrow[cust_id]
        for key, fval in fetched.items():
            gval = got[key]
            if fval is None:
                assert gval is None, f"{key}[{cust_id}] should stay null"
            elif isinstance(fval, (int, float, Decimal)):
                # numeric aggregates come back from the cursor as Decimal but the
                # Arrow path casts numeric -> float64 (numeric_as_float default);
                # compare as floats, not string reprs.
                assert float(gval) == pytest.approx(float(fval)), key
            else:
                assert str(gval) == str(fval), key


def test_parquet_round_trip_preserves_nulls(pg_conn, tmp_path: Path):
    """Write Parquet, reload with pyarrow, assert values + Arrow-null fidelity."""
    _seed_two_customers(pg_conn)
    featurizer = _featurizer(_config())

    table = featurizer.to_arrow(connection=pg_conn)
    out = tmp_path / "features.parquet"
    pq.write_table(table, out)

    reloaded = pq.read_table(out)
    assert reloaded.schema.equals(table.schema)
    assert reloaded.to_pydict() == table.to_pydict()

    # the no-orders row's measures are still Parquet/Arrow nulls after the trip
    d = reloaded.to_pydict()
    row2 = d["customer_id"].index(2)
    assert d["MEAN(orders.amount)"][row2] is None
    assert reloaded.column("MEAN(orders.amount)").null_count == 1


def test_to_parquet_method_writes_file(pg_conn, tmp_path: Path):
    """Featurizer.to_parquet writes a readable Parquet file."""
    _seed_two_customers(pg_conn)
    out = tmp_path / "features.parquet"
    _featurizer(_config()).to_parquet(str(out), connection=pg_conn)

    assert out.exists()
    reloaded = pq.read_table(out)
    assert reloaded.num_rows == 2
    assert "customer_id" in reloaded.column_names


def test_arrow_imputation_contract_on_export(pg_conn):
    """impute=True on the Arrow path: count->0, measures null, __missing flags."""
    _seed_two_customers(pg_conn)
    table = _featurizer(_config()).to_arrow(connection=pg_conn, impute=True)

    d = table.to_pydict()
    row2 = d["customer_id"].index(2)

    # missing flags present and set for the no-orders row
    assert d["COUNT(orders.order_id)__missing"][row2] == 1
    assert d["MEAN(orders.amount)__missing"][row2] == 1

    # count-like filled to 0; measures left null
    assert d["COUNT(orders.order_id)"][row2] == 0
    assert d["SUM(orders.amount)"][row2] == 0
    assert d["MEAN(orders.amount)"][row2] is None


def test_arrow_mean_strategy_gated(pg_conn):
    """Leaky mean strategy is refused without the opt-in and warns with it."""
    _seed_two_customers(pg_conn)
    featurizer = _featurizer(_config())

    with pytest.raises(ValueError, match="temporal leakage"):
        featurizer.to_arrow(connection=pg_conn, impute=True, measure_strategy="mean")

    with pytest.warns(UserWarning, match="temporal leakage"):
        table = featurizer.to_arrow(
            connection=pg_conn,
            impute=True,
            measure_strategy="mean",
            allow_full_matrix_fit=True,
        )
    # with a single populated row, the mean fills the no-orders measure
    d = table.to_pydict()
    row2 = d["customer_id"].index(2)
    assert d["MEAN(orders.amount)"][row2] is not None
