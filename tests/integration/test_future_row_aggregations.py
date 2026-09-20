"""No aggregation reads a child row dated after the as-of date (matrix cell 10).

The transformer side of this question found six leaks (issue #27). The
aggregation side was measured clean on c2c21a7 — 67 of 67 — and had no test to
keep it so: ``tests/primitives/test_causal_safety.py`` asserts that the causal
fragment is *rendered*, which is not the same as the value not moving.

Same method as ``test_asof_bounded_child_read.py``: run each registered
aggregation twice, the second time with child rows dated after the as-of date
(one the day after, one months later), with one interval and the whole history.
A point-in-time-correct matrix cannot tell the runs apart.
"""

from __future__ import annotations

import os

import pytest

from featurizer import Featurizer
from featurizer.primitives.utils import list_aggregations
from tests.test_aggregation_quoting import (
    AS_OF,
    ROWS,
    SPATIAL,
    SQL_TYPE,
    STAMPS,
    _config,
    _input_type,
    _literal,
)

pytestmark = pytest.mark.integration

# Lower than, and different from, every knowable value, so it moves every
# statistic it is allowed to touch.
FUTURE = {"numeric": 0.25, "categorical": "c", "boolean": False}
FUTURE_STAMPS = ("2024-06-02", "2024-09-01")


def _matrix(query: str, vtype: str, *, future_rows: bool) -> dict:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    rows, event_id = [], 0
    for series_id in (1, 2):
        for stamp, value in zip(STAMPS, ROWS[vtype]):
            event_id += 1
            rows.append(
                f"({event_id}, {series_id}, date '{stamp}', {_literal(value)}, "
                f"{19.0 + event_id / 10}, {-99.0 - event_id / 7})"
            )
    if future_rows:
        for offset, stamp in enumerate(FUTURE_STAMPS, start=1):
            rows.append(
                f"({event_id + offset}, 1, date '{stamp}', "
                f"{_literal(FUTURE[vtype])}, 25.5, -80.25)"
            )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values (1), (2)")
        cur.execute(
            "create temp table events (event_id int, series_id int, ts date, "
            f"x {SQL_TYPE[vtype]}, lat double precision, lon double precision)"
        )
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
def test_a_row_after_the_as_of_date_moves_no_aggregation(tmp_path, name) -> None:
    vtype = _input_type(name)
    column_type = "numeric" if vtype == "index" else vtype
    latlon = ("lat", "lon") if name in SPATIAL else None
    featurizer = Featurizer(_config(tmp_path, "x", [name], column_type, latlon=latlon))
    assert featurizer.feature_manifest, f"{name} emitted no feature to check"
    query = featurizer.query
    knowable = _matrix(query, column_type, future_rows=False)
    with_future = _matrix(query, column_type, future_rows=True)
    assert with_future == knowable
