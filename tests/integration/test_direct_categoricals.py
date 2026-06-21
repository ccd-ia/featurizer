"""End-to-end direct-categorical one-hot encoding against a real PostgreSQL.

The fixture mirrors the **updated DirtyDuck food-inspections schema** that triage
ships (``~/projects/triage/dirtyduck/food_db``, the raw/clean/ontology rework that
added PostgreSQL ENUMs "to give featurizer a fixed one-hot vocabulary for free").
It is seeded inline on the rolled-back ``pg_conn`` transaction — no network, no
``just seed`` — and is faithful to triage's real types:

- Low-cardinality categoricals are PostgreSQL ENUMs: ``risk_t`` (low/medium/high)
  and ``result_t`` (7 labels), exactly as triage defines them in
  ``02_create_cleaned_inspections_table.sql``. These are featurizer's fixed,
  ENUM-introspected one-hot vocabulary.
- ``facility_type`` stays **text** (287 distinct in the real data) — high
  cardinality, deliberately *not* an ENUM; the consumer's train-fit encoder owns
  it, so featurizer excludes it (``role: identifier``) or fails loud if asked to
  one-hot it without a declared vocabulary.

The target is ``events`` (one row per inspection, ``ontology.events`` shape):
``risk`` / ``result`` are its direct ENUM categoricals.
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
    """Create the ``dirtyduck`` schema (triage's ENUMs + events) on ``conn``.

    Everything lives inside the caller's transaction, so the ``pg_conn``
    rollback discards it; the target database is left untouched.
    """
    with conn.cursor() as cur:
        cur.execute("drop schema if exists dirtyduck cascade")
        cur.execute("create schema dirtyduck")
        # Triage's real clean-layer ENUMs (verbatim labels + order).
        cur.execute("create type dirtyduck.risk_t as enum ('low', 'medium', 'high')")
        cur.execute(
            "create type dirtyduck.result_t as enum "
            "('pass', 'pass w/ conditions', 'fail', 'no entry', 'not ready', "
            "'out of business', 'business not located')"
        )
        cur.execute(
            "create type dirtyduck.inspection_type_t as enum "
            "('canvass', 'task force', 'complaint', 'food poisoning', "
            "'consultation', 'license', 'tag removal')"
        )
        cur.execute("""
            create table dirtyduck.events (
                event_id      integer primary key,
                entity_id     bigint,
                date          date,
                type          dirtyduck.inspection_type_t,
                risk          dirtyduck.risk_t,
                result        dirtyduck.result_t,
                -- denormalized point-in-time state; high-cardinality TEXT (not an ENUM)
                facility_type text
            )
            """)
        cur.executemany(
            "insert into dirtyduck.events "
            "(event_id, entity_id, date, type, risk, result, facility_type) "
            "values (%s, %s, %s, %s, %s, %s, %s)",
            [
                (1, 10, "2015-01-05", "canvass", "high", "fail", "restaurant"),
                (2, 10, "2015-02-10", "complaint", "high", "pass", "restaurant"),
                (3, 11, "2015-03-15", "canvass", "low", "pass", "grocery store"),
                (
                    4,
                    12,
                    "2015-04-20",
                    "license",
                    "medium",
                    "pass w/ conditions",
                    "school",
                ),
                (5, 13, "2015-05-25", "canvass", "high", "fail", "bakery"),
                (
                    6,
                    14,
                    "2015-06-30",
                    "complaint",
                    "medium",
                    "no entry",
                    "mobile food dispenser",
                ),
                # NULL risk -> an all-zero risk one-hot row (never a crash).
                (7, 15, "2015-07-04", "canvass", None, "out of business", "restaurant"),
            ],
        )


def _config() -> dict:
    return {
        "target": "events",
        "max_depth": 1,
        "intervals": [],
        "aggregations": [],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "events",
                "id": "event_id",
                "table": "dirtyduck.events",
                "temporal_ix": "date",
                "variables": {
                    # ENUM-typed -> introspected vocabulary, no declared list
                    "risk": {"type": "categorical", "role": "categorical"},
                    "result": {"type": "categorical", "role": "categorical"},
                    # high-cardinality TEXT -> excluded (the consumer's encoder owns it)
                    "facility_type": {"type": "text", "role": "identifier"},
                },
            }
        ],
        "relationships": [],
    }


def _write(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return handle.name


# Sorted ENUM labels -> deterministic one-hot column order.
_RISK_COLUMNS = [
    "events.risk=high",
    "events.risk=low",
    "events.risk=medium",
]
_RESULT_COLUMNS = [
    "events.result=business not located",
    "events.result=fail",
    "events.result=no entry",
    "events.result=not ready",
    "events.result=out of business",
    "events.result=pass",
    "events.result=pass w/ conditions",
]


def test_enum_sourced_one_hot_end_to_end(pg_conn) -> None:
    pytest.importorskip("pyarrow")
    _seed_dirtyduck(pg_conn)
    make_as_of_dates(pg_conn, ["2016-01-01"])

    featurizer = Featurizer(_write(_config()), connection=pg_conn)

    # Vocabulary is read from each column's PostgreSQL ENUM (no declared list).
    # The exact set is the two ENUMs' labels; column order is deterministic
    # (features sort by their quoted identifier — covered in the DB-free unit
    # test; here membership is what matters).
    one_hot_cols = [
        e.column for e in featurizer.feature_manifest if e.kind == "one_hot"
    ]
    assert set(one_hot_cols) == set(_RISK_COLUMNS + _RESULT_COLUMNS)

    table = featurizer.to_arrow(connection=pg_conn)
    names = table.column_names

    # The high-cardinality text column never reaches the output.
    assert "facility_type" not in names
    for col in _RISK_COLUMNS + _RESULT_COLUMNS:
        assert col in names

    rows: Dict[int, Dict[str, Any]] = {
        row["event_id"]: row for row in table.to_pylist()
    }

    # event 1: risk=high, result=fail -> exactly those indicators are 1.
    assert rows[1]["events.risk=high"] == 1
    assert rows[1]["events.risk=low"] == 0
    assert rows[1]["events.risk=medium"] == 0
    assert rows[1]["events.result=fail"] == 1
    assert rows[1]["events.result=pass"] == 0
    # event 4: result with a space/special label resolves cleanly.
    assert rows[4]["events.result=pass w/ conditions"] == 1

    # NULL risk -> an all-zero risk one-hot row (never a crash).
    assert all(rows[7][col] == 0 for col in _RISK_COLUMNS)
    # ...but its (non-null) result is still encoded.
    assert rows[7]["events.result=out of business"] == 1


def test_one_hot_columns_survive_impute(pg_conn) -> None:
    pytest.importorskip("pyarrow")
    _seed_dirtyduck(pg_conn)
    make_as_of_dates(pg_conn, ["2016-01-01"])

    featurizer = Featurizer(_write(_config()), connection=pg_conn)
    imputed = featurizer.to_arrow(connection=pg_conn, impute=True)
    names = imputed.column_names

    # One-hots are never NULL, so imputation emits no ``__missing`` flag for them.
    for col in _RISK_COLUMNS + _RESULT_COLUMNS:
        assert col in names
        assert f"{col}__missing" not in names


def test_high_cardinality_text_categorical_without_vocabulary_fails_loud(
    pg_conn,
) -> None:
    """A text (non-ENUM) column asked to one-hot without a vocabulary fails loud.

    This is the boundary triage relies on: ``facility_type`` is deliberately text
    (high cardinality), so featurizer refuses to invent a vocabulary for it — the
    consumer's train-fit encoder owns it. Introspection finds no ENUM and the
    error tells the user to declare a vocabulary or type the column as an ENUM.
    """
    _seed_dirtyduck(pg_conn)
    config = _config()
    config["entities"][0]["variables"]["facility_type"] = {
        "type": "text",
        "role": "categorical",  # ask to one-hot a text column with no vocabulary
    }
    with pytest.raises(ValueError, match="vocabulary|ENUM"):
        Featurizer(_write(config), connection=pg_conn)
