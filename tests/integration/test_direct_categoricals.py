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


# --------------------------------------------------------------------------- #
# Parent-entity categoricals (ADR-0019). The target ``inspections`` is the
# child; its parent ``facilities`` owns the categorical. The indicators are
# emitted under the parent's alias and, deprecated for one minor release,
# under the target's alias, which is what releases up to 1.3 emitted.
# --------------------------------------------------------------------------- #

PARENT_AS_OF = ["2016-01-01"]

# facility 1 changes type on 2015-06-01; the 2017 row is dated after the
# as-of date and may move nothing.
_FACILITY_HISTORY = [
    (1, "2015-01-01", "restaurant", "small"),
    (1, "2015-06-01", "grocery", "large"),
    (2, "2015-01-01", "O'Brien's", "small"),
    (3, "2015-01-01", None, None),
]
_FACILITY_AFTER_AS_OF = [(1, "2017-01-01", "restaurant", "small")]
_FACILITY_KNOWABLE_CHANGE = [(1, "2015-02-01", "grocery", "large")]

_INSPECTIONS = [
    (100, 1, "2015-03-01", 90),
    (101, 1, "2015-07-01", 80),
    (102, 2, "2015-03-01", 70),
    (103, 3, "2015-03-01", 60),  # NULL type -> all-zero
    (104, 4, "2015-03-01", 50),  # no facility -> all-zero
]

_VOCABULARY = ["grocery", "restaurant", "O'Brien's"]


def _seed_parent(conn: Any, facility_rows: list) -> None:
    with conn.cursor() as cur:
        cur.execute("drop schema if exists parentduck cascade")
        cur.execute("create schema parentduck")
        cur.execute("create type parentduck.size_t as enum ('small', 'large')")
        cur.execute(
            "create table parentduck.facilities (facility_id int, valid_from date, "
            "facility_type text, size_class parentduck.size_t)"
        )
        cur.execute(
            "create table parentduck.inspections "
            "(inspection_id int, facility_id int, date date, score numeric)"
        )
        cur.executemany(
            "insert into parentduck.inspections values (%s, %s, %s, %s)",
            _INSPECTIONS,
        )
    _insert_facilities(conn, facility_rows)
    make_as_of_dates(conn, PARENT_AS_OF)


def _insert_facilities(conn: Any, facility_rows: list) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "insert into parentduck.facilities values (%s, %s, %s, %s)",
            facility_rows,
        )


def _parent_config(*, as_of: bool, parent_alias: str = "facilities") -> dict:
    relationship: Dict[str, Any] = {
        "parent": {"entity": parent_alias, "key": "facility_id"},
        "child": {"entity": "inspections", "key": "facility_id"},
    }
    if as_of:
        relationship["temporal"] = {"mode": "as_of"}
    return {
        "target": "inspections",
        "max_depth": 2,
        "intervals": [],
        "aggregations": [],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "inspections",
                "id": "inspection_id",
                "table": "parentduck.inspections",
                "temporal_ix": "date",
                "variables": {"score": {"type": "numeric"}},
            },
            {
                "alias": parent_alias,
                "id": "facility_id",
                "table": "parentduck.facilities",
                "temporal_ix": "valid_from",
                "variables": {
                    "facility_type": {
                        "type": "categorical",
                        "role": "categorical",
                        "vocabulary": _VOCABULARY,
                    },
                    # ENUM-typed, no declared list: introspected on the parent.
                    "size_class": {"type": "categorical", "role": "categorical"},
                },
            },
        ],
        "relationships": [relationship],
    }


# inspection -> (grocery, restaurant, O'Brien's, large, small), hand-computed
# from the row of facilities in force at the inspection's date.
_EXPECTED_AS_OF = {
    100: (0, 1, 0, 0, 1),  # 2015-01-01 row: restaurant, small
    101: (1, 0, 0, 1, 0),  # 2015-06-01 row: grocery, large
    102: (0, 0, 1, 0, 1),  # O'Brien's (a quote in the value), small
    103: (0, 0, 0, 0, 0),  # NULL -> all-zero
    104: (0, 0, 0, 0, 0),  # no parent row -> all-zero
}


def _indicator_columns(alias: str) -> list:
    return [
        f"{alias}.facility_type=grocery",
        f"{alias}.facility_type=restaurant",
        f"{alias}.facility_type=O'Brien's",
        f"{alias}.size_class=large",
        f"{alias}.size_class=small",
    ]


def _parent_rows(conn: Any, config: dict) -> Dict[int, Dict[str, Any]]:
    featurizer = Featurizer(_write(config), connection=conn)
    frame = featurizer.to_dataframe(connection=conn).reset_index()
    return {int(row["inspection_id"]): row for row in frame.to_dict("records")}


@pytest.mark.parametrize("parent_alias", ["facilities", "FacSites"])
def test_parent_one_hot_reads_the_row_in_force_at_the_target_date(
    pg_conn, parent_alias
) -> None:
    """Both name families, hand-computed, under ``temporal: {mode: as_of}``.

    ``FacSites`` is an alias that is not lower case: it only appears inside the
    quoted output name.
    """
    _seed_parent(pg_conn, _FACILITY_HISTORY)
    rows = _parent_rows(pg_conn, _parent_config(as_of=True, parent_alias=parent_alias))
    for family in (parent_alias, "inspections"):
        columns = _indicator_columns(family)
        got = {key: tuple(int(row[c]) for c in columns) for key, row in rows.items()}
        assert got == _EXPECTED_AS_OF, family


def test_parent_one_hot_without_a_temporal_block_reads_the_static_join(
    pg_conn,
) -> None:
    # One row per facility, so the static join does not fan out.
    _seed_parent(pg_conn, [r for r in _FACILITY_HISTORY if r[1] == "2015-01-01"])
    rows = _parent_rows(pg_conn, _parent_config(as_of=False))
    columns = _indicator_columns("facilities")
    got = {key: tuple(int(rows[key][c]) for c in columns) for key in rows}
    assert got == {
        100: (0, 1, 0, 0, 1),
        101: (0, 1, 0, 0, 1),
        102: (0, 0, 1, 0, 1),
        103: (0, 0, 0, 0, 0),
        104: (0, 0, 0, 0, 0),
    }
    for key, row in rows.items():
        for new, old in zip(columns, _indicator_columns("inspections")):
            assert row[new] == row[old], (key, new)


def test_a_parent_row_dated_after_the_as_of_date_moves_nothing(pg_conn) -> None:
    _seed_parent(pg_conn, _FACILITY_HISTORY)
    config = _parent_config(as_of=True)
    knowable = _parent_rows(pg_conn, config)

    _insert_facilities(pg_conn, _FACILITY_AFTER_AS_OF)
    assert _parent_rows(pg_conn, config) == knowable

    # The control: a row dated before inspection 100 is in force at its date,
    # so it does move that inspection, under both names.
    _insert_facilities(pg_conn, _FACILITY_KNOWABLE_CHANGE)
    moved = _parent_rows(pg_conn, config)
    for family in ("facilities", "inspections"):
        assert knowable[100][f"{family}.facility_type=restaurant"] == 1
        assert moved[100][f"{family}.facility_type=restaurant"] == 0
        assert moved[100][f"{family}.facility_type=grocery"] == 1


def _three_paths(conn: Any, config: dict) -> Dict[str, Dict[tuple, Dict[str, Any]]]:
    from featurizer.sharding import ColumnGroupSharder

    path = _write(config)

    def run(sql: str) -> list:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [d.name for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def rejoin(queries: Any) -> Dict[tuple, Dict[str, Any]]:
        joined: Dict[tuple, Dict[str, Any]] = {}
        for sql in queries.values():
            for row in run(sql):
                key = (row["as_of_date"], row["inspection_id"])
                joined.setdefault(key, {}).update(row)
        return joined

    single = {
        (row["as_of_date"], row["inspection_id"]): row
        for row in run(Featurizer(path, connection=conn).query)
    }
    built = ColumnGroupSharder(
        Featurizer(path, connection=conn)._plan, max_columns_per_group=4
    ).build()
    assert len(built.queries) > 1, "expected the small budget to force >1 group"
    grouped = Featurizer(path, connection=conn, materialize_threshold=1)._grouped()
    assert grouped.materialization is not None, "expected a TEMP-table preamble"
    with conn.cursor() as cur:
        for ddl in grouped.materialization.ddl:
            cur.execute(ddl)
    return {
        "single query": single,
        "column groups": rejoin(built.queries),
        "TEMP tables": rejoin(grouped.queries),
    }


@pytest.mark.parametrize("as_of", [True, False], ids=["as_of", "static"])
def test_the_three_render_paths_agree_on_parent_one_hot(pg_conn, as_of) -> None:
    history = _FACILITY_HISTORY
    if not as_of:
        history = [r for r in _FACILITY_HISTORY if r[1] == "2015-01-01"]
    _seed_parent(pg_conn, history)
    paths = _three_paths(pg_conn, _parent_config(as_of=as_of))
    single = paths["single query"]
    assert any(
        value == 1
        for row in single.values()
        for column, value in row.items()
        if column.startswith("facilities.")
    )
    assert paths["column groups"] == single
    assert paths["TEMP tables"] == single
