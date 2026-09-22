"""A reduction over a degenerate series returns NULL, not rounding noise (#67).

``cosinor_amplitude_weekly`` regresses the value on the weekly sine and cosine
of the timestamp. Rows a whole number of weeks apart share a phase, so the basis
is constant up to the rounding of ``sin()`` at an argument near 1e4 radians, and
``regr_slope`` divides by that rounding: prices in the hundreds gave amplitudes
around 1e15, and a different one on every read. With a ``date`` index every row
sits on one of seven phases, so a series on one weekday is not exotic.

The sweep is over every aggregation with a numeric input: on a series whose
timestamps all share the weekly phase, each returns NULL or a number of the
data's magnitude. The one primitive with a period is the one that failed; the
net is for the next one.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from featurizer import Featurizer
from featurizer.primitives.utils import list_aggregations
from tests.test_aggregation_quoting import _config, _input_type

pytestmark = pytest.mark.integration

AS_OF = "2014-06-01"
# Whole weeks apart: one phase. Two series, so a cross-series statistic has a
# second group to look at. These five dates give BOTH basis columns a variance
# that is rounding noise rather than an exact zero (4e-25 and 7e-25 on
# PostgreSQL 16); a set where either is exactly zero returns NULL on its own and
# proves nothing.
WEEKLY = ["2013-03-01", "2013-05-10", "2013-08-02", "2013-11-15", "2014-01-10"]
VALUES = [828.7, 2268.0, 435.64, 248.69, 1190.5]
# Above any statistic of the data (values to 4,536; a variance is 2.6e6), well
# below the noise-driven 1e15.
MAGNITUDE = 1e9


def _run(query: str, *, off_phase: int = 0) -> dict:
    """``off_phase`` moves that many rows of each series one day later, off the
    shared phase."""
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    rows, event_id = [], 0
    for series_id in (1, 2):
        for position, (stamp, value) in enumerate(zip(WEEKLY, VALUES)):
            event_id += 1
            shift = " + 1" if position < off_phase else ""
            rows.append(
                f"({event_id}, {series_id}, date '{stamp}'{shift}, "
                f"{value * series_id}, {19.0 + event_id / 10}, {-99.0 - event_id / 7})"
            )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values (1), (2)")
        cur.execute(
            "create temp table events (event_id int, series_id int, ts date, "
            "x double precision, lat double precision, lon double precision)"
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


NUMERIC = sorted(name for name in list_aggregations() if _input_type(name) == "numeric")


@pytest.mark.parametrize("name", NUMERIC)
def test_a_series_on_one_weekly_phase_gives_null_or_a_value_of_the_datas_size(
    tmp_path, name
) -> None:
    featurizer = Featurizer(_config(tmp_path, "x", [name]), validate=False)
    assert featurizer.feature_manifest, f"{name} emitted no feature to check"
    for series_id, row in _run(featurizer.query).items():
        for column, value in row.items():
            if column in ("as_of_date", "series_id") or value is None:
                continue
            if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                assert abs(float(value)) < MAGNITUDE, (series_id, column, value)


def test_cosinor_is_null_on_one_phase_and_finite_with_a_second(tmp_path) -> None:
    featurizer = Featurizer(
        _config(tmp_path, "x", ["cosinor_amplitude_weekly"]), validate=False
    )
    column = "COSINOR_AMPLITUDE_WEEKLY(events.x)"
    on_one_phase = _run(featurizer.query)
    assert on_one_phase[1][column] is None
    assert on_one_phase[2][column] is None

    # Two rows a day off the phase: the basis has spread, and the amplitude is
    # a number of the data's size on both series.
    spread = _run(featurizer.query, off_phase=2)
    for series_id in (1, 2):
        value = spread[series_id][column]
        assert value is not None
        assert 0 < float(value) < MAGNITUDE, (series_id, value)
