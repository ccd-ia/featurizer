"""No transformer on a child entity reads a row dated after the as-of date (#27).

The method is the one the issue measured with: run a config twice, the second
time with one more child row dated AFTER the as-of date. A point-in-time-correct
matrix cannot tell the two runs apart.

Before the child read was bounded, six transformers could: ``percent_rank`` and
``ntile`` (they divide by the partition's size), ``last`` (an explicit
``unbounded following`` frame), ``cusum`` (``avg(x) over (partition by id)``),
and ``cross_entity_zscore`` / ``cross_entity_percentile`` (``over ()`` spans
every child row, so these two leaked even when the child ``id`` is unique per
row). ``cdf`` would have been the seventh, had ``cum_dist()`` existed.

The sweep runs every registered transformer, so the next whole-partition window
somebody registers is covered the day it lands.
"""

from __future__ import annotations

import statistics

import pytest

from featurizer.primitives.utils import get_transformers, list_transformations

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration

AS_OF = "2024-06-01"
# One row a month, the last one dated ON the as-of date (the boundary is
# inclusive by default, so it is knowable).
PAST = {
    "numeric": [1.0, 2.0, 0.5, 0.7, 3.0, 1.2],
    "date": [
        "2023-01-05",
        "2023-02-15",
        "2023-03-31",
        "2023-04-02",
        "2023-05-09",
        "2023-05-30",
    ],
    "text": ["alpha beta", "Gamma!", "delta, epsilon.", "a", "bb", "ccc"],
    "categorical": ["a", "b", "a", "b", "a", "b"],
    "boolean": [True, False, True, True, False, True],
}
# Lower than every past value, so it moves every rank it is allowed to touch.
FUTURE = {
    "numeric": 0.1,
    "date": "2022-12-25",
    "text": "",
    "categorical": "c",
    "boolean": False,
}
SQL_TYPE = {
    "numeric": "double precision",
    "date": "date",
    "text": "text",
    "categorical": "text",
    "boolean": "boolean",
}
# Not configurable as a bare transformations: entry.
NOT_STANDALONE = {"identity", "in_array"}


def _config(transformer: str, vtype: str, child_id: str) -> dict:
    return {
        "target": "series",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["max", "min", "nunique", "count"],
        "transformations": ["identity", transformer],
        "entities": [
            {"alias": "series", "table": "series", "id": "series_id"},
            {
                "alias": "events",
                "table": "events",
                "id": child_id,
                "temporal_ix": "ts",
                "variables": {"x": {"type": vtype}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_id"},
            }
        ],
    }


def _seed(conn, vtype: str, *, future_row: bool) -> None:
    create_temp_table(conn, "series", [("series_id", "int")], [(1,), (2,)])
    rows = []
    for series_id in (1, 2):
        for month, value in enumerate(PAST[vtype], start=1):
            rows.append((len(rows) + 1, series_id, f"2024-{month:02d}-01", value))
    if future_row:
        rows.append((len(rows) + 1, 1, "2024-09-01", FUTURE[vtype]))
    create_temp_table(
        conn,
        "events",
        [
            ("event_id", "int"),
            ("series_id", "int"),
            ("ts", "date"),
            ("x", SQL_TYPE[vtype]),
        ],
        rows,
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [(AS_OF,)])


def _matrix(conn, config: dict, vtype: str, *, future_row: bool) -> dict:
    """Run inside a savepoint so the two worlds never share a temp table."""
    with conn.transaction(force_rollback=True):
        _seed(conn, vtype, future_row=future_row)
        return {row["series_id"]: row for row in run_featurizer(conn, config)}


def _sweep_cases():
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
        # ``series_id``: several rows per partition, the event-source shape the
        # issue names. ``event_id``: one row per partition, which still leaked
        # through the two ``over ()`` transformers.
        for child_id in ("series_id", "event_id"):
            yield pytest.param(name, vtype, child_id, id=f"{name}-by-{child_id}")


@pytest.mark.parametrize("name,vtype,child_id", list(_sweep_cases()))
def test_a_row_after_the_as_of_date_moves_nothing(
    pg_conn, name, vtype, child_id
) -> None:
    config = _config(name, vtype, child_id)
    knowable = _matrix(pg_conn, config, vtype, future_row=False)
    with_future = _matrix(pg_conn, config, vtype, future_row=True)
    assert with_future == knowable


# ------------------------------------------------------------------ values
# What each of the seven returns once it can only see knowable rows, computed
# here from PAST alone. The future row is present in every one of these runs.

X = PAST["numeric"]


def _value(pg_conn, transformer: str, agg: str) -> float:
    config = _config(transformer, "numeric", "series_id")
    row = _matrix(pg_conn, config, "numeric", future_row=True)[1]
    return float(row[f"{agg}(events.{transformer.upper()}(events.x))"])


def test_percent_rank_divides_by_the_knowable_rows(pg_conn) -> None:
    # Ordered by ts, so the row ON the as-of date is the last of six: (6-1)/(6-1).
    # With the future row counted it was 5/6.
    assert _value(pg_conn, "percent_rank", "MAX") == 1.0


def test_cdf_executes_and_divides_by_the_knowable_rows(pg_conn) -> None:
    assert _value(pg_conn, "cdf", "MAX") == 1.0
    assert _value(pg_conn, "cdf", "MIN") == pytest.approx(1 / 6)


def test_ntile_spreads_the_knowable_rows(pg_conn) -> None:
    # Six rows over five tiles reach tile 5; seven rows left the sixth in tile 4.
    assert _value(pg_conn, "ntile", "MAX") == 5


def test_last_is_the_last_knowable_row(pg_conn) -> None:
    assert _value(pg_conn, "last", "MAX") == X[-1]
    assert _value(pg_conn, "last", "MIN") == X[-1]


def test_cusum_centres_on_the_knowable_mean(pg_conn) -> None:
    mean = statistics.fmean(X)
    running = [sum(X[: k + 1]) - (k + 1) * mean for k in range(len(X))]
    assert _value(pg_conn, "cusum", "MAX") == pytest.approx(max(running))
    assert _value(pg_conn, "cusum", "MIN") == pytest.approx(min(running))


def test_cross_entity_zscore_standardizes_over_the_knowable_rows(pg_conn) -> None:
    population = X * 2  # both series carry the same six values
    mean, stdev = statistics.fmean(population), statistics.stdev(population)
    assert _value(pg_conn, "cross_entity_zscore", "MAX") == pytest.approx(
        (max(X) - mean) / stdev
    )


def test_cross_entity_percentile_ranks_among_the_knowable_rows(pg_conn) -> None:
    # 12 rows; the two 3.0s tie at rank 11: (11-1)/(12-1).
    assert _value(pg_conn, "cross_entity_percentile", "MAX") == pytest.approx(10 / 11)
