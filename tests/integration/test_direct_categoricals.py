"""End-to-end direct-categorical one-hot encoding against a real PostgreSQL.

A small, deterministic "DirtyDuck" food-inspections fixture (the same dataset
shape triage-pg validates against) is seeded inline on the rolled-back
``pg_conn`` transaction — no network, no ``just seed``. It carries a target
``facilities`` entity with ``facility_type`` typed as a PostgreSQL ``ENUM`` and a
``name`` identifier column, plus an ``inspections`` child with ``result`` / ``risk``
enums. The test drives the ENUM-introspection vocabulary path (no declared
vocabulary) all the way through ``to_arrow``.
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict

import pytest
import yaml

from featurizer import Featurizer

from ._realistic import make_as_of_dates

pytestmark = pytest.mark.integration


def _seed_dirtyduck(conn: Any) -> None:
    """Create the ``dirtyduck`` schema (enums + tables + rows) on ``conn``.

    Everything lives inside the caller's transaction, so the ``pg_conn``
    rollback discards it; the target database is left untouched.
    """
    with conn.cursor() as cur:
        cur.execute("drop schema if exists dirtyduck cascade")
        cur.execute("create schema dirtyduck")
        cur.execute(
            "create type dirtyduck.facility_type_enum as enum "
            "('Restaurant', 'Grocery Store', 'School', 'Bakery')"
        )
        cur.execute(
            "create type dirtyduck.inspection_result as enum "
            "('Pass', 'Fail', 'Pass w/ Conditions')"
        )
        cur.execute(
            "create type dirtyduck.risk_level as enum "
            "('Risk 1 (High)', 'Risk 2 (Medium)', 'Risk 3 (Low)')"
        )
        cur.execute("""
            create table dirtyduck.facilities (
                license_no    bigint primary key,
                name          text,
                facility_type dirtyduck.facility_type_enum,
                risk          dirtyduck.risk_level,
                first_seen    date
            )
            """)
        cur.execute("""
            create table dirtyduck.inspections (
                inspection_id   bigint primary key,
                license_no      bigint,
                inspection_date date,
                result          dirtyduck.inspection_result,
                risk            dirtyduck.risk_level
            )
            """)
        cur.executemany(
            "insert into dirtyduck.facilities "
            "(license_no, name, facility_type, risk, first_seen) "
            "values (%s, %s, %s, %s, %s)",
            [
                (1, "Alpha Cafe", "Restaurant", "Risk 1 (High)", "2014-01-05"),
                (2, "Beta Mart", "Grocery Store", "Risk 2 (Medium)", "2014-02-10"),
                (3, "Gamma School", "School", "Risk 3 (Low)", "2014-03-15"),
                (4, "Delta Diner", "Restaurant", "Risk 1 (High)", "2014-04-20"),
                (5, "Epsilon Bakehouse", "Bakery", "Risk 2 (Medium)", "2014-05-25"),
                # NULL facility_type -> must yield an all-zero one-hot row.
                (6, "Zeta Unknown", None, "Risk 3 (Low)", "2014-06-30"),
            ],
        )
        cur.executemany(
            "insert into dirtyduck.inspections "
            "(inspection_id, license_no, inspection_date, result, risk) "
            "values (%s, %s, %s, %s, %s)",
            [
                (101, 1, "2014-07-01", "Pass", "Risk 1 (High)"),
                (102, 1, "2014-08-01", "Fail", "Risk 1 (High)"),
                (103, 2, "2014-07-15", "Pass", "Risk 2 (Medium)"),
                (104, 6, "2014-07-20", "Pass w/ Conditions", "Risk 3 (Low)"),
            ],
        )


def _config() -> dict:
    return {
        "target": "facilities",
        "max_depth": 2,
        "intervals": ["P1Y"],
        "aggregations": ["count"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "facilities",
                "id": "license_no",
                "table": "dirtyduck.facilities",
                "temporal_ix": "first_seen",
                "variables": {
                    # identifier -> excluded from output
                    "name": {"type": "text", "role": "identifier"},
                    # categorical, NO declared vocabulary -> read from the ENUM
                    "facility_type": {"type": "categorical", "role": "categorical"},
                },
            },
            {
                "alias": "inspections",
                "id": "inspection_id",
                "table": "dirtyduck.inspections",
                "temporal_ix": "inspection_date",
                "variables": {"license_no": {"type": "index"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "facilities", "key": "license_no"},
                "child": {"entity": "inspections", "key": "license_no"},
            }
        ],
    }


def _write(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return handle.name


# Sorted ENUM labels -> deterministic one-hot column order.
_ONE_HOT_COLUMNS = [
    "facilities.facility_type=Bakery",
    "facilities.facility_type=Grocery Store",
    "facilities.facility_type=Restaurant",
    "facilities.facility_type=School",
]


def test_enum_sourced_one_hot_end_to_end(pg_conn) -> None:
    pytest.importorskip("pyarrow")
    _seed_dirtyduck(pg_conn)
    make_as_of_dates(pg_conn, ["2015-01-01"])

    featurizer = Featurizer(_write(_config()), connection=pg_conn)

    # Vocabulary is read from the column's PostgreSQL ENUM (no declared list),
    # sorted for a deterministic one-hot column order.
    one_hot_cols = [
        e.column for e in featurizer.feature_manifest if e.kind == "one_hot"
    ]
    assert one_hot_cols == _ONE_HOT_COLUMNS

    table = featurizer.to_arrow(connection=pg_conn)
    names = table.column_names

    # The identifier column never reaches the output.
    assert "name" not in names
    for col in _ONE_HOT_COLUMNS:
        assert col in names

    rows: Dict[int, Dict[str, Any]] = {
        row["license_no"]: row for row in table.to_pylist()
    }

    # A Restaurant facility -> exactly its indicator is 1, the rest 0.
    assert rows[1]["facilities.facility_type=Restaurant"] == 1
    assert rows[1]["facilities.facility_type=Bakery"] == 0
    assert rows[1]["facilities.facility_type=School"] == 0
    assert rows[5]["facilities.facility_type=Bakery"] == 1

    # A NULL facility_type -> an all-zero one-hot row (never a crash).
    assert all(rows[6][col] == 0 for col in _ONE_HOT_COLUMNS)

    # The child event stream still aggregates to a numeric feature.
    count_cols = [n for n in names if n.startswith("COUNT(")]
    assert count_cols, "expected a COUNT aggregate over inspections"


def test_one_hot_columns_survive_impute(pg_conn) -> None:
    pytest.importorskip("pyarrow")
    _seed_dirtyduck(pg_conn)
    make_as_of_dates(pg_conn, ["2015-01-01"])

    featurizer = Featurizer(_write(_config()), connection=pg_conn)
    imputed = featurizer.to_arrow(connection=pg_conn, impute=True)
    names = imputed.column_names

    # One-hots are never NULL, so imputation emits no ``__missing`` flag for them.
    for col in _ONE_HOT_COLUMNS:
        assert col in names
        assert f"{col}__missing" not in names
