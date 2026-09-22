"""A value does not depend on the physical order of the rows (issue #66, ADR-0018).

A primitive that walks a timeline orders its rows by the temporal index. Two rows
of one partition on the same timestamp have no order, PostgreSQL returns them as
they arrive, and a lag, a transition or an autocorrelation depends on which came
first. Measured on the live databases with the plain dense query, the same rows
stored in another physical order: 28 columns on 25 of dirtyduck's entities, 36
columns on 1,437 of donorschoose's 3,000, every one of them with a tie. A reload,
a ``cluster`` or a different plan above the scan is enough to move them.

The sweep: every registered primitive, the same rows inserted forwards and
backwards into a TEMP table, equal values. Each tied pair differs in the input
column, which is what the tiebreak falls back to; two rows equal in every column
the ordering reads are interchangeable, and need no order.

Transformers are observed per ROW, on an event-like target, because an aggregate
over their output (``max``, ``count``) hides most of what moved.
"""

from __future__ import annotations

import os
import random
from decimal import Decimal

import pytest
import yaml

from featurizer import Featurizer
from featurizer.primitives.utils import (
    get_transformers,
    list_aggregations,
    list_transformations,
)
from tests.test_aggregation_quoting import SPATIAL, _input_type
from tests.test_aggregation_quoting import _config as _aggregation_config

pytestmark = pytest.mark.integration

AS_OF = "2024-06-01"
# Three ties per series: rows 1-2, 4-5 and 7-8 share a date.
STAMPS = [
    "2024-01-05",
    "2024-01-20",
    "2024-01-20",
    "2024-02-14",
    "2024-03-03",
    "2024-03-03",
    "2024-04-11",
    "2024-05-25",
    "2024-05-25",
]
SERIES = (1, 2, 3, 4, 5, 6)
SQL_TYPE = {
    "numeric": "double precision",
    "date": "date",
    "text": "text",
    "categorical": "text",
    "boolean": "boolean",
}
# Not configurable as a bare ``transformations:`` entry.
NOT_STANDALONE = {"identity", "in_array"}

# States by series and position. Series 1 and 2 are written by hand for the two
# statistics a symmetric pattern hides: in series 1 the first tie sits between an
# ``a`` and a ``b``, so its order decides whether the longest streak is 2
# (a a b b) or 1 (a b a b); in series 2 it decides when the current state began.
# The other four are drawn once from a fixed seed, so that a primitive nobody
# wrote a pattern for still meets a tie that matters.
_RNG = random.Random(66)
STATES = {
    1: "aabbcdeab",
    2: "abaaaaaaa",
    **{s: "".join(_RNG.choice("abc") for _ in STAMPS) for s in (3, 4, 5, 6)},
}
NUMBERS = {s: [round(_RNG.uniform(0.2, 2.8), 3) for _ in STAMPS] for s in SERIES}
# Series 2's second tie is two rows equal in x as well (its states already
# are): the same base row twice, as dirtyduck's inspections have. A run-length
# built from two sorts numbered such a pair differently in each.
NUMBERS[2][5] = NUMBERS[2][4]
# A second variable, distinct on every row except that pair, and read by no
# primitive under test: the row it tells apart must keep its own value.
OTHER = {s: [round(10 * s + i * 1.5, 1) for i, _ in enumerate(STAMPS)] for s in SERIES}
OTHER[2][5] = OTHER[2][4]


def _value(vtype: str, series_id: int, position: int):
    """Two rows of a tie that are equal in the column a primitive reads are
    interchangeable for it; every other pair has to come out in one order."""
    n = series_id * 10 + position
    return {
        "numeric": NUMBERS[series_id][position],
        "date": f"2023-{(n % 12) + 1:02d}-{(n % 27) + 1:02d}",
        "text": f"word{n} tail{series_id}, x{position}.",
        "categorical": STATES[series_id][position],
        "boolean": STATES[series_id][position] == "a",
    }[vtype]


def _literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        return float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-9)
    return a == b


def _row_key(row) -> tuple:
    """A row as a comparable tuple: floats to nine significant digits, because a
    population statistic (``avg(x) over ()``) is a float sum, and a float sum
    depends on the order of its terms in its last bits."""
    out = []
    for value in row:
        if isinstance(value, bool) or value is None:
            out.append(str(value))
        elif isinstance(value, (int, float, Decimal)):
            out.append(f"{float(value):.9g}")
        else:
            out.append(str(value))
    return tuple(out)


def _connect():
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    return psycopg.connect(url)


def _event_rows(vtype: str, series: tuple[int, ...]) -> list[str]:
    rows, event_id = [], 0
    for position, stamp in enumerate(STAMPS):
        for series_id in series:
            event_id += 1
            rows.append(
                f"({event_id}, {series_id}, date '{stamp}', "
                f"{_literal(_value(vtype, series_id, position))}, "
                f"{OTHER[series_id][position]}, "
                f"{19.0 + event_id / 10}, {-99.0 - event_id / 7})"
            )
    return rows


EVENTS_DDL = (
    "create temp table events (event_id int, series_id int, ts date, "
    "x {x}, y double precision, lat double precision, lon double precision)"
)


def _assert_equal(forwards: dict, backwards: dict) -> None:
    assert forwards, "the query returned nothing to compare"
    assert set(forwards) == set(backwards)
    moved = sorted(
        {
            column
            for key in forwards
            for column in forwards[key]
            if not _same(forwards[key][column], backwards[key][column])
        }
    )
    assert moved == [], f"depends on the physical order of tied rows: {moved}"


# ------------------------------------------------------------- aggregations


def _aggregate(query: str, vtype: str, *, reverse: bool) -> dict:
    rows = _event_rows(vtype, SERIES)
    if reverse:
        rows.reverse()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values " + ", ".join(f"({s})" for s in SERIES))
        cur.execute(EVENTS_DDL.format(x=SQL_TYPE[vtype]))
        cur.execute(f"insert into events values {', '.join(rows)}")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute(f"insert into as_of_dates values (date '{AS_OF}')")
        cur.execute(query)
        names = [d.name for d in cur.description]
        out = {
            row[names.index("series_id")]: dict(zip(names, row))
            for row in cur.fetchall()
        }
        conn.rollback()
    return out


@pytest.mark.parametrize("name", sorted(list_aggregations()))
def test_no_aggregation_depends_on_the_order_of_tied_rows(tmp_path, name) -> None:
    vtype = _input_type(name)
    column_type = "numeric" if vtype == "index" else vtype
    latlon = ("lat", "lon") if name in SPATIAL else None
    featurizer = Featurizer(
        _aggregation_config(tmp_path, "x", [name], column_type, latlon=latlon),
        validate=False,
    )
    assert featurizer.feature_manifest, f"{name} emitted no feature to check"
    query = featurizer.query
    _assert_equal(
        _aggregate(query, column_type, reverse=False),
        _aggregate(query, column_type, reverse=True),
    )


# ------------------------------------------------------------- transformers


def _transformer_config(name: str, vtype: str) -> dict:
    """An event-like target: several rows per ``series_id``, so a window has a
    partition to walk, and its rows ARE the output. No unique identifier on
    purpose: the row order has to come from the declared variables."""
    return {
        "target": "events",
        "max_depth": 1,
        "intervals": [],
        "aggregations": [],
        "transformations": ["identity", name],
        "entities": [
            {
                "alias": "events",
                "table": "events",
                "id": "series_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": vtype}, "y": {"type": "numeric"}},
            }
        ],
    }


def _transform(query: str, vtype: str, *, reverse: bool) -> list:
    """The output rows as a sorted list: a row is identified by everything it
    carries, so two identical base rows may trade places and nothing else may."""
    rows = _event_rows(vtype, SERIES)
    if reverse:
        rows.reverse()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(EVENTS_DDL.format(x=SQL_TYPE[vtype]))
        cur.execute(f"insert into events values {', '.join(rows)}")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute(f"insert into as_of_dates values (date '{AS_OF}')")
        cur.execute(query)
        out = sorted(_row_key(row) for row in cur.fetchall())
        conn.rollback()
    return out


def _transformer_cases():
    for name in sorted(list_transformations()):
        if name in NOT_STANDALONE:
            continue
        types = getattr(get_transformers([name])[name], "input_types", None) or [
            "numeric"
        ]
        vtype = (
            "numeric"
            if "numeric" in types
            else next((t for t in types if t in SQL_TYPE), None)
        )
        if vtype is None:
            continue
        yield pytest.param(name, vtype, id=name)


@pytest.mark.parametrize("name,vtype", list(_transformer_cases()))
def test_no_transformer_depends_on_the_order_of_tied_rows(
    tmp_path, name, vtype
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_transformer_config(name, vtype), sort_keys=False))
    query = Featurizer(str(path), validate=False).query
    forwards = _transform(query, vtype, reverse=False)
    assert len(forwards) == len(SERIES) * len(STAMPS)
    backwards = _transform(query, vtype, reverse=True)
    moved = [a for a, b in zip(forwards, backwards) if a != b]
    assert moved == [], f"{len(moved)} rows depend on the physical order: {moved[:2]}"


# ----------------------------------------------------------- as-of lookup


def test_an_as_of_lookup_takes_the_same_source_row_in_either_order(tmp_path) -> None:
    """``limit 1`` over "the most recent source row": two source rows on one
    timestamp differ in the value looked up, and the row taken must not depend
    on the physical order of the source table."""
    config = {
        "target": "series",
        "max_depth": 3,
        "intervals": [],
        "aggregations": ["max", "min", "sum"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "series", "table": "series", "id": "series_id"},
            {
                "alias": "events",
                "table": "events",
                "id": "event_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": "numeric"}, "zone": {"type": "index"}},
            },
            {
                "alias": "rates",
                "table": "rates",
                "temporal_ix": "valid_from",
                "variables": {"rate": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_id"},
            },
            {
                "parent": {"entity": "rates", "key": "zone"},
                "child": {"entity": "events", "key": "zone"},
                "temporal": {"mode": "as_of"},
            },
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    query = Featurizer(str(path), validate=False).query
    # Zone 1 has three rates on 2024-01-01 and two on 2024-03-01; no row of the
    # source is unique in anything but its value.
    rates = [
        "(1, date '2024-01-01', 0.10)",
        "(1, date '2024-01-01', 0.50)",
        "(1, date '2024-01-01', 0.30)",
        "(1, date '2024-03-01', 0.70)",
        "(1, date '2024-03-01', 0.20)",
        "(2, date '2024-01-01', 0.90)",
    ]

    def run(reverse: bool) -> list:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("create temp table series (series_id int)")
            cur.execute("insert into series values (1), (2)")
            cur.execute(
                "create temp table events (event_id int, series_id int, zone int, "
                "ts date, x double precision)"
            )
            cur.execute(
                "insert into events values (1, 1, 1, '2024-02-10', 1.0), "
                "(2, 1, 1, '2024-04-10', 2.0), (3, 2, 2, '2024-02-10', 3.0), "
                "(4, 2, 1, '2024-05-10', 4.0)"
            )
            cur.execute(
                "create temp table rates (zone int, valid_from date, rate numeric)"
            )
            cur.execute(
                "insert into rates values "
                + ", ".join(reversed(rates) if reverse else rates)
            )
            cur.execute("create temp table as_of_dates (as_of_date date)")
            cur.execute(f"insert into as_of_dates values (date '{AS_OF}')")
            cur.execute(query)
            out = sorted(_row_key(row) for row in cur.fetchall())
            conn.rollback()
        return out

    forwards = run(reverse=False)
    assert forwards
    assert forwards == run(reverse=True)
