"""Imputation × oversized-child temp-table materialization, combined
(v1.0 hardening).

Both features existed and were tested separately; this file exercises them
*together*: a config whose child chain is forced through the issue-#7
TEMP-table materialization preamble, executed via ``to_arrow(impute=True)``
and ``to_dataframe(impute=True)``. The imputation contract must hold
unchanged on the materialized path — count-like features fill with the
structural 0, measures keep NULL plus a ``__missing`` indicator, and every
other value equals the unimputed run.
"""

from __future__ import annotations

import datetime
import math
import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.imputation import MISSING_INDICATOR_SUFFIX

from ._harness import create_temp_table

pa = pytest.importorskip("pyarrow")

pytestmark = pytest.mark.integration


def _config() -> dict:
    """stores <- orders <- items; store 2 has no orders, so every feature of
    store 2 is NULL — the imputation surface."""
    return {
        "target": "stores",
        "max_depth": 3,
        "intervals": [],
        "aggregations": ["count", "sum", "mean"],
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


def _seed(conn) -> None:
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


def _materialized_featurizer() -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(_config(), handle)
        path = handle.name
    # threshold=1 forces the whole non-target child chain into TEMP shards.
    return Featurizer(path, validate=False, materialize_threshold=1)


def _is_count_like(name: str) -> bool:
    return name.startswith(("COUNT(", "SUM(", "NUNIQUE("))


def _null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def test_to_arrow_impute_on_materialized_path(pg_conn):
    _seed(pg_conn)
    f = _materialized_featurizer()
    assert f._grouped().materialization is not None, "expected the #7 preamble"

    plain = f.to_arrow(connection=pg_conn)
    imputed = f.to_arrow(connection=pg_conn, impute=True)
    assert isinstance(plain, pa.Table) and isinstance(imputed, pa.Table)

    prows = {r["store_id"]: r for r in plain.to_pylist()}
    irows = {r["store_id"]: r for r in imputed.to_pylist()}
    feature_cols = [
        n for n in plain.column_names if n not in ("as_of_date", "store_id")
    ]
    assert feature_cols

    # Store 2 (no orders): counts fill with the structural 0, measures stay
    # NULL; a __missing indicator appears for every column that had NULLs.
    for col in feature_cols:
        assert _null(prows[2][col]), f"precondition: {col} should be NULL for store 2"
        indicator = f"{col}{MISSING_INDICATOR_SUFFIX}"
        assert indicator in imputed.column_names, f"missing indicator for {col}"
        assert irows[2][indicator] == 1
        assert irows[1][indicator] == 0
        if _is_count_like(col):
            assert irows[2][col] == 0, f"{col} should fill with structural 0"
        else:
            assert _null(irows[2][col]), f"measure {col} must stay NULL"

    # Store 1 (has orders): every value equals the unimputed run.
    for col in feature_cols:
        assert irows[1][col] == prows[1][col], f"{col} changed for store 1"


def test_to_dataframe_impute_on_materialized_path(pg_conn):
    _seed(pg_conn)
    f = _materialized_featurizer()

    plain = f.to_dataframe(connection=pg_conn)
    imputed = f.to_dataframe(connection=pg_conn, impute=True)

    assert list(imputed.index.names) == ["as_of_date", "store_id"]
    feature_cols = [c for c in plain.columns]
    indicator_cols = [
        c for c in imputed.columns if c.endswith(MISSING_INDICATOR_SUFFIX)
    ]
    assert indicator_cols

    p1 = plain.xs(1, level="store_id").iloc[0]
    p2 = plain.xs(2, level="store_id").iloc[0]
    i1 = imputed.xs(1, level="store_id").iloc[0]
    i2 = imputed.xs(2, level="store_id").iloc[0]

    for col in feature_cols:
        assert _null(p2[col]), f"precondition: {col} should be NULL for store 2"
        if _is_count_like(col):
            assert i2[col] == 0
        else:
            assert _null(i2[col]), f"measure {col} must stay NULL"
        # Store 1 values are untouched by imputation.
        assert (_null(i1[col]) and _null(p1[col])) or i1[col] == p1[col]

    for col in indicator_cols:
        base = col[: -len(MISSING_INDICATOR_SUFFIX)]
        assert i2[col] == 1 and i1[col] == 0, f"indicator {col} wrong"
        assert base in feature_cols


def test_arrow_and_dataframe_agree_on_materialized_impute(pg_conn):
    """The two imputed engine paths tell the same story (same columns, same
    fills) — the Arrow path is not a second implementation drifting away."""
    _seed(pg_conn)
    f = _materialized_featurizer()

    arrow_rows = {
        r["store_id"]: r
        for r in f.to_arrow(connection=pg_conn, impute=True).to_pylist()
    }
    frame = f.to_dataframe(connection=pg_conn, impute=True)

    frame_cols = set(frame.columns)
    arrow_cols = set(next(iter(arrow_rows.values()))) - {"as_of_date", "store_id"}
    assert arrow_cols == frame_cols

    for sid in (1, 2):
        frow = frame.xs(sid, level="store_id").iloc[0]
        arow = arrow_rows[sid]
        for col in frame_cols:
            fv, av = frow[col], arow[col]
            assert (_null(fv) and _null(av)) or float(fv) == float(
                av
            ), f"{col} differs between Arrow and DataFrame for store {sid}"
