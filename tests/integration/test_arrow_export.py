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
from loguru import logger

from featurizer import Featurizer
from featurizer.arrow import ArrowExporter

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


# ------------------------------------------------------------------ #
# #95: one execution, bounded batches, output independent of batch size.
# ------------------------------------------------------------------ #

#: Every OID the exporter maps, NULLs in most columns, one numeric that fits
#: decimal128(38, 9) and one that does not, and one unmapped OID (interval).
_TYPES_QUERY = """
select current_date + (g % 3) as as_of_date, g::int8 as entity_id,
  case when g % 3 = 0 then null else g::int4 end as i4,
  (g % 100)::int2 as i2,
  case when g % 5 = 0 then null else (g / 7.0)::float4 end as f4,
  case when g % 6 = 0 then null else g / 3.0::float8 end as f8,
  case when g % 4 = 0 then null else (g * 1.5)::numeric end as num_fits,
  case when g % 4 = 0 then null else g::numeric / 3 end as num_overflows,
  case when g % 2 = 0 then null else 'é' || g end as txt,
  ('c' || (g % 9))::char(3) as bp,
  ('v' || g)::varchar as vc,
  g % 2 = 0 as b,
  '2020-01-01'::timestamp + g * interval '1 hour' as ts,
  '2020-01-01 00:00+00'::timestamptz + g * interval '1 minute' as tstz,
  case when g % 10 = 0 then null else md5(g::text)::uuid end as u,
  case when g % 7 = 0 then null else g * interval '1 day' end as iv,
  null::int8 as all_null
from generate_series(1, 50) as g
"""


def test_the_query_runs_once(pg_conn, tmp_path: Path):
    """Column names and types come from a describe: COPY is the only execution."""
    with pg_conn.cursor() as cur:
        cur.execute("create temp sequence executions")
    query = (
        "select current_date as as_of_date, nextval('executions') as entity_id "
        "from generate_series(1, 10)"
    )
    ArrowExporter().to_arrow(query, connection=pg_conn)
    ArrowExporter().write_parquet(
        query, str(tmp_path / "m.parquet"), connection=pg_conn
    )

    with pg_conn.cursor() as cur:
        cur.execute("select last_value from executions")
        (calls,) = cur.fetchone()
    assert calls == 20, "10 rows x 2 exports; 40 means each export ran twice"


@pytest.mark.parametrize("numeric_as_float", [True, False])
def test_batch_size_changes_chunking_only(pg_conn, tmp_path: Path, numeric_as_float):
    """One row per batch and one batch for everything build the same table and file."""
    kwargs = {"connection": pg_conn, "numeric_as_float": numeric_as_float}
    per_row = ArrowExporter(batch_values=1).to_arrow(_TYPES_QUERY, **kwargs)
    whole = ArrowExporter(batch_values=10**9).to_arrow(_TYPES_QUERY, **kwargs)

    assert per_row.column("i4").num_chunks == 50
    assert whole.column("i4").num_chunks == 1
    assert per_row.schema.equals(whole.schema)
    assert per_row.equals(whole)

    for batch_values in (1, 10**9):
        out = tmp_path / f"m{batch_values}.parquet"
        ArrowExporter(batch_values=batch_values).write_parquet(
            _TYPES_QUERY, str(out), **kwargs
        )
        assert pq.read_table(out).equals(whole)


def test_types_and_nulls_of_the_batched_decode(pg_conn):
    """OID-derived types, Arrow nulls (never NaN), uuid as its string."""
    table = ArrowExporter(batch_values=7).to_arrow(_TYPES_QUERY, connection=pg_conn)
    types = {f.name: f.type for f in table.schema}

    assert types["as_of_date"] == pa.date32()
    assert types["entity_id"] == pa.int64()
    assert types["i4"] == pa.int32() and types["i2"] == pa.int16()
    assert types["f4"] == pa.float32() and types["f8"] == pa.float64()
    assert types["num_fits"] == pa.float64() and types["num_overflows"] == pa.float64()
    assert types["txt"] == types["bp"] == types["vc"] == types["u"] == pa.string()
    assert types["b"] == pa.bool_()
    assert types["ts"] == pa.timestamp("us")
    assert types["tstz"] == pa.timestamp("us", tz="UTC")
    assert types["all_null"] == pa.int64()
    assert pa.types.is_interval(types["iv"]) or pa.types.is_duration(types["iv"])

    d = table.to_pydict()
    assert table.column("f8").null_count == 50 // 6
    assert d["f8"][5] is None  # g = 6, not NaN
    assert table.column("all_null").null_count == 50
    with pg_conn.cursor() as cur:
        cur.execute("select md5('1')::uuid::text")
        (first_uuid,) = cur.fetchone()
    assert d["u"][0] == first_uuid


def test_exact_numeric_keeps_decimal_or_warns_to_string(pg_conn):
    """numeric_as_float=False: decimal128 where every value fits, else string + warning."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        table = ArrowExporter(batch_values=7).to_arrow(
            _TYPES_QUERY, connection=pg_conn, numeric_as_float=False
        )
    finally:
        logger.remove(sink)

    assert table.schema.field("num_fits").type == pa.decimal128(38, 9)
    assert table.column("num_fits").to_pylist()[0] == Decimal("1.5")
    assert table.schema.field("num_overflows").type == pa.string()
    with pg_conn.cursor() as cur:
        cur.execute("select (1::numeric / 3)::text")
        (one_third,) = cur.fetchone()
    assert table.column("num_overflows").to_pylist()[0] == one_third
    assert table.column("num_overflows").null_count == 50 // 4
    fallbacks = [m for m in messages if "falling back to string" in m]
    assert len(fallbacks) == 1 and "'num_overflows'" in fallbacks[0]


def test_a_failed_streaming_write_leaves_no_file(pg_conn, tmp_path: Path):
    """A query that fails mid-COPY leaves neither the target nor a partial file."""
    out = tmp_path / "m.parquet"
    failing = (
        "select current_date as as_of_date, g as entity_id, 1 / (g - 40) as boom "
        "from generate_series(1, 50) as g"
    )
    with pytest.raises(RuntimeError, match="division by zero"):
        ArrowExporter(batch_values=3).write_parquet(
            failing, str(out), connection=pg_conn
        )
    assert list(tmp_path.iterdir()) == []
