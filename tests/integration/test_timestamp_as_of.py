"""An interval window cuts at the as-of instant, on a timestamp (issue #87).

Every as-of test before #87 built ``date`` columns and a ``date`` as-of date,
and moved its future rows by whole days. On that shape the window's old
``daterange(...) @> ts::date`` and the causal cut ``ts <= aod.as_of_date``
select the same rows, so no test could see that on a ``timestamp`` they do not:
under ``exclusive`` with a ``timestamp`` as-of date the window dropped the whole
as-of day (P30D counted 1 row where the whole history counted 3), and its lower
bound reached back to midnight (P1D counted 42 hours).

ADR-0021 writes the window on the column at both bounds, its upper bound the
causal predicate. This file holds the issue's hand cases, with the values each
shape must give, and a registry-wide sweep on a ``timestamp`` child with rows
stamped within hours of a ``timestamp`` as-of date, in both boundary modes:

1. rows outside the window, measured at the instant, move no interval column;
2. when every row is inside the window, an interval column equals the
   whole-history column (``WINDOW_IS_NOT_WHOLE_HISTORY`` lists the
   aggregations whose interval variant is a different statistic, measured).
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from featurizer import Featurizer
from featurizer.primitives.aggregations import TwoWindowDriftAggregator
from featurizer.primitives.utils import get_aggregations, list_aggregations
from tests.test_aggregation_quoting import (
    ROWS,
    SPATIAL,
    SQL_TYPE,
    _config,
    _input_type,
    _literal,
)

pytestmark = pytest.mark.integration

MODES = ("inclusive", "exclusive")


def _connect():
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    return psycopg.connect(url)


def _write(tmp_path: Path, config: str, boundary: str) -> str:
    """``_config``'s file with the boundary mode appended (a top-level key)."""
    path = Path(config)
    path.write_text(path.read_text() + f"as_of_boundary: {boundary}\n")
    return str(path)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


# ------------------------------------------------------------- hand cases

# The issue's five rows: the day before, then 00:00 / 12:00 / 23:59:59 ON the
# as-of day, then 00:00 the day after. x is a power of ten so a sum names rows.
EVENTS = [
    (1, "2024-05-31 12:00:00", 1.0),
    (2, "2024-06-01 00:00:00", 10.0),
    (3, "2024-06-01 12:00:00", 100.0),
    (4, "2024-06-01 23:59:59", 1000.0),
    (5, "2024-06-02 00:00:00", 10000.0),
]

# (column type, as-of date type, as-of date, session TimeZone, mode) ->
# (COUNT, SUM) for (whole history, P1D, P30D).
HAND_CASES = {
    # A date as-of date is 00:00 of its day: rows 3 and 4 are knowable from
    # the next as-of date. Unchanged by #87.
    ("timestamp", "date", "2024-06-01", "UTC", "inclusive"): [
        (2, 11.0),
        (2, 11.0),
        (2, 11.0),
    ],
    ("timestamp", "date", "2024-06-01", "UTC", "exclusive"): [
        (1, 1.0),
        (1, 1.0),
        (1, 1.0),
    ],
    # A timestamp as-of date: P1D is the 24 hours before 18:00, which leaves
    # row 1 (30 hours before) out. Before #87 the inclusive P1D counted it (3,
    # 111), and the exclusive windows dropped rows 2 and 3 (1, 1.0).
    ("timestamp", "timestamp", "2024-06-01 18:00:00", "UTC", "inclusive"): [
        (3, 111.0),
        (2, 110.0),
        (3, 111.0),
    ],
    ("timestamp", "timestamp", "2024-06-01 18:00:00", "UTC", "exclusive"): [
        (3, 111.0),
        (2, 110.0),
        (3, 111.0),
    ],
    # On a date column every row of the as-of day is stamped 00:00. Unchanged.
    ("date", "date", "2024-06-01", "UTC", "inclusive"): [
        (4, 1111.0),
        (4, 1111.0),
        (4, 1111.0),
    ],
    ("date", "date", "2024-06-01", "UTC", "exclusive"): [
        (1, 1.0),
        (1, 1.0),
        (1, 1.0),
    ],
    # timestamptz: the literals and the date's midnight are both read in the
    # session TimeZone, so a non-UTC zone gives the timestamp column's values.
    ("timestamptz", "date", "2024-06-01", "America/Mexico_City", "inclusive"): [
        (2, 11.0),
        (2, 11.0),
        (2, 11.0),
    ],
    ("timestamptz", "date", "2024-06-01", "America/Mexico_City", "exclusive"): [
        (1, 1.0),
        (1, 1.0),
        (1, 1.0),
    ],
}


def _hand_config(tmp_path: Path, boundary: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        "target: series\n"
        "max_depth: 2\n"
        "intervals: [P1D, P30D]\n"
        "aggregations: [count, sum]\n"
        "transformations: [identity]\n"
        f"as_of_boundary: {boundary}\n"
        "entities:\n"
        "  - alias: series\n"
        "    id: series_id\n"
        "    table: series\n"
        "  - alias: events\n"
        "    id: event_id\n"
        "    table: events\n"
        "    temporal_ix: ts\n"
        "    variables:\n"
        "      x: {type: numeric}\n"
        "relationships:\n"
        "  - parent: {entity: series, key: series_id}\n"
        "    child: {entity: events, key: series_id}\n"
    )
    return str(path)


@pytest.mark.parametrize(
    ("ts_type", "aod_type", "aod", "zone", "mode"),
    list(HAND_CASES),
    ids=[f"{k[0]}-{k[1]}-{k[3]}-{k[4]}" for k in HAND_CASES],
)
def test_the_issue_rows_give_the_instant_values(
    tmp_path, ts_type: str, aod_type: str, aod: str, zone: str, mode: str
) -> None:
    query = Featurizer(_hand_config(tmp_path, mode)).query
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(f"set timezone = '{zone}'")
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values (1)")
        cur.execute(
            f"create temp table events (event_id int, series_id int, ts {ts_type}, "
            "x double precision)"
        )
        for event_id, stamp, x in EVENTS:
            cur.execute(
                f"insert into events values (%s, 1, %s::{ts_type}, %s)",
                (event_id, stamp, x),
            )
        cur.execute(f"create temp table as_of_dates (as_of_date {aod_type})")
        cur.execute(f"insert into as_of_dates values (%s::{aod_type})", (aod,))
        cur.execute(query)
        names = [d.name for d in cur.description]
        row = dict(zip(names, cur.fetchone()))
        conn.rollback()
    got = [
        (row[f"COUNT(events.event_id{w})"], row[f"SUM(events.x{w})"])
        for w in ("", "|interval=P1D", "|interval=P30D")
    ]
    expected = HAND_CASES[(ts_type, aod_type, aod, zone, mode)]
    assert [(int(c), float(s or 0)) for c, s in got] == expected


# ------------------------------------------------------- registry-wide sweep

AS_OF = datetime(2024, 6, 1, 18, 0)
WINDOW = timedelta(days=1)
INTERVAL = "P1D"

# Hours from the as-of instant. The window is [-24, 0]; the drift baseline is
# [-48, -24). -30 sits on the window's lower calendar day, which the old
# day-granular bound let in; +2 sits on the as-of day after the instant; 0 is
# the instant itself, inside under ``inclusive`` only.
OFFSETS = [-54, -40, -36, -30, -24, -20, -16, -10, -4, 0, 2, 6]

# The interval variant is a different statistic from the whole-history one,
# so (2) does not apply. Measured on this file's data; each reason is why.
WINDOW_IS_NOT_WHOLE_HISTORY: dict[str, str] = {}


def _is_drift(name: str) -> bool:
    return isinstance(get_aggregations([name])[name], TwoWindowDriftAggregator)


def _inside(offset: int, mode: str, *, windows: int) -> bool:
    """Is the row ``offset`` hours from the as-of instant inside ``windows``
    intervals back from the cut, under ``mode``?"""
    stamp = AS_OF + timedelta(hours=offset)
    lower = AS_OF - windows * WINDOW
    upper = stamp <= AS_OF if mode == "inclusive" else stamp < AS_OF
    return lower <= stamp and upper


def _sweep_matrix(query: str, vtype: str, keep) -> dict:
    values = ROWS[vtype]
    rows, event_id = [], 0
    for series_id in (1, 2):
        for i, offset in enumerate(OFFSETS):
            event_id += 1
            if not keep(offset):
                continue
            stamp = (AS_OF + timedelta(hours=offset)).isoformat(sep=" ")
            rows.append(
                f"({event_id}, {series_id}, timestamp '{stamp}', "
                f"{_literal(values[(i + series_id) % len(values)])}, "
                f"{19.0 + event_id / 10}, {-99.0 - event_id / 7})"
            )
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values (1), (2)")
        cur.execute(
            "create temp table events (event_id int, series_id int, ts timestamp, "
            f"x {SQL_TYPE[vtype]}, lat double precision, lon double precision)"
        )
        cur.execute(f"insert into events values {', '.join(rows)}")
        cur.execute("create temp table as_of_dates (as_of_date timestamp)")
        cur.execute(
            "insert into as_of_dates values (%s::timestamp)",
            (AS_OF.isoformat(sep=" "),),
        )
        cur.execute(query)
        names = [d.name for d in cur.description]
        out = {
            row[names.index("series_id")]: dict(zip(names, row))
            for row in cur.fetchall()
        }
        conn.rollback()
    return out


def _sweep_query(tmp_path: Path, name: str, mode: str) -> tuple[str, str]:
    vtype = _input_type(name)
    column_type = "numeric" if vtype == "index" else vtype
    latlon = ("lat", "lon") if name in SPATIAL else None
    config = _config(
        tmp_path, "x", [name], column_type, intervals=(INTERVAL,), latlon=latlon
    )
    featurizer = Featurizer(_write(tmp_path, config, mode))
    assert featurizer.feature_manifest, f"{name} emitted no feature to check"
    return featurizer.query, column_type


def _interval_columns(matrix: dict) -> list[str]:
    row = next(iter(matrix.values()))
    return [c for c in row if f"|interval={INTERVAL}" in c]


def _cases(names) -> list:
    return [pytest.param(n, m, id=f"{n}-{m}") for n in sorted(names) for m in MODES]


def _outside_cases() -> list:
    return _cases(list_aggregations())


def _whole_history_cases() -> list:
    return _cases(n for n in list_aggregations() if not _is_drift(n))


@pytest.mark.parametrize(("name", "mode"), _outside_cases())
def test_a_row_outside_the_instant_window_moves_no_interval_column(
    tmp_path, name: str, mode: str
) -> None:
    query, vtype = _sweep_query(tmp_path, name, mode)
    windows = 2 if _is_drift(name) else 1
    every_row = _sweep_matrix(query, vtype, lambda offset: True)
    inside = _sweep_matrix(
        query, vtype, lambda offset: _inside(offset, mode, windows=windows)
    )
    columns = _interval_columns(every_row)
    assert columns, f"{name} emitted no {INTERVAL} column"
    moved = {
        (series, column): (every_row[series][column], inside[series][column])
        for series in every_row
        for column in columns
        if not _same(every_row[series][column], inside[series][column])
    }
    assert not moved, f"{name} under {mode}: rows outside the window moved {moved}"


@pytest.mark.parametrize(("name", "mode"), _whole_history_cases())
def test_an_interval_holding_every_row_equals_the_whole_history(
    tmp_path, name: str, mode: str
) -> None:
    if name in WINDOW_IS_NOT_WHOLE_HISTORY:
        pytest.skip(WINDOW_IS_NOT_WHOLE_HISTORY[name])
    query, vtype = _sweep_query(tmp_path, name, mode)
    matrix = _sweep_matrix(
        query, vtype, lambda offset: _inside(offset, mode, windows=1)
    )
    moved = {}
    for column in _interval_columns(matrix):
        whole = column.replace(f"|interval={INTERVAL}", "")
        for series, row in matrix.items():
            assert whole in row, f"{name}: no whole-history column for {column}"
            if not _same(row[column], row[whole]):
                moved[(series, column)] = (row[column], row[whole])
    assert not moved, f"{name} under {mode}: interval != whole history {moved}"
