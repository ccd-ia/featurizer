"""No planner pass reads a row dated after the as-of date (matrix cell 11).

The peer-group, spatial and graph-relationship passes and the as-of lookup each
carry their own causal cut — they read base tables, so the cut on a child's
read (issue #27) does not cover them — and none had a test that adds an
unknowable row and looks at the values. This one does, for every table a pass
reads: the target itself (a later peer), the right table of a spatial
relationship, the edge table, the neighbour-state entity, and the lookup's
source.

Two configs: the three passes hang off the target, the lookup off a child that
the target then aggregates (under an interval and ``count`` too, since issue
#48).
"""

from __future__ import annotations

import datetime

import pytest

from ._harness import run_featurizer

pytestmark = pytest.mark.integration

AS_OF = [datetime.date(2024, 6, 1), datetime.date(2024, 8, 1)]


def _passes_config() -> dict:
    return {
        "target": "series",
        "max_depth": 2,
        "intervals": ["P90D"],
        "aggregations": ["max", "count"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "series",
                "table": "series",
                "id": "series_id",
                "temporal_ix": "opened",
                "spatial_ix": {"lat": "lat", "lon": "lon"},
                "variables": {
                    "grp": {"type": "categorical"},
                    "size": {"type": "numeric"},
                },
                "peer_groups": [{"by": "grp", "measures": ["size"]}],
            },
            {
                "alias": "events",
                "table": "events",
                "id": "event_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": "numeric"}},
            },
            {
                "alias": "sites",
                "table": "sites",
                "id": "site_id",
                "temporal_ix": "site_ts",
                "spatial_ix": {"lat": "site_lat", "lon": "site_lon"},
            },
            {
                "alias": "states",
                "table": "states",
                "id": "state_id",
                "temporal_ix": "state_ts",
                "variables": {
                    "risk": {"type": "numeric"},
                    "flagged": {"type": "boolean"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_ref"},
            }
        ],
        "spatial_relationships": [
            {
                "name": "near_sites",
                "left": "series",
                "right": "sites",
                "within_m": 5000,
                "bandwidth_m": 1000,
            }
        ],
        "graph_relationships": [
            {
                "name": "links",
                "left": "series",
                "right": "states",
                "edges": {
                    "table": "links",
                    "source": "src",
                    "target": "dst",
                    "timestamp": "linked",
                },
                "directed": False,
            }
        ],
    }


def _lookup_config() -> dict:
    return {
        "target": "series",
        "max_depth": 3,
        "intervals": ["P90D"],
        "aggregations": ["max", "count"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "series", "table": "series", "id": "series_id"},
            {
                "alias": "events",
                "table": "events",
                "id": "event_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": "numeric"}},
            },
            {
                "alias": "rates",
                "table": "rates",
                "id": "rate_id",
                "temporal_ix": "rate_ts",
                "variables": {"level": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_ref"},
            },
            {
                "parent": {"entity": "rates", "key": "zone_id"},
                "child": {"entity": "events", "key": "zone_ref"},
                "temporal": {"mode": "as_of"},
            },
        ],
    }


# (table, ddl columns, knowable rows, rows dated after the last as-of date)
TABLES = [
    (
        "series",
        "series_id int, opened date, lat double precision, lon double precision, "
        "grp text, size double precision",
        "(1, '2024-01-01', 19.40, -99.10, 'a', 10.0), "
        "(2, '2024-01-05', 19.41, -99.11, 'a', 20.0), "
        "(3, '2024-02-01', 19.42, -99.12, 'a', 40.0), "
        "(4, '2024-07-01', 19.43, -99.13, 'a', 80.0)",
        # a later peer of the same group, right next to series 1
        "(9, '2024-09-10', 19.40, -99.10, 'a', 1000.0)",
    ),
    (
        "events",
        "event_id int, series_ref int, zone_ref int, ts date, x double precision",
        "(10, 1, 7, '2024-03-10', 1.5), (11, 1, 7, '2024-05-20', 4.0), "
        "(12, 2, 8, '2024-05-01', 6.0), (13, 3, 7, '2024-07-15', 2.0)",
        "(19, 1, 7, '2024-09-30', 99.0)",
    ),
    (
        "sites",
        "site_id int, site_ts date, site_lat double precision, "
        "site_lon double precision",
        "(1, '2024-01-01', 19.401, -99.101), (2, '2024-07-01', 19.412, -99.109)",
        "(9, '2024-09-01', 19.400, -99.100)",
    ),
    (
        "states",
        "state_id int, state_ts date, risk double precision, flagged boolean",
        "(1, '2024-01-01', 0.1, true), (2, '2024-01-01', 0.4, false), "
        "(3, '2024-01-01', 0.8, true)",
        # a later state of a node that is somebody's neighbour
        "(2, '2024-09-01', 50.0, true)",
    ),
    (
        "links",
        "src int, dst int, linked date",
        "(1, 2, '2024-02-01'), (1, 3, '2024-05-01'), (2, 3, '2024-07-10')",
        "(1, 2, '2024-09-15'), (3, 2, '2024-09-16')",
    ),
    (
        "rates",
        "rate_id int, zone_id int, rate_ts date, level double precision",
        "(1, 7, '2024-01-01', 0.5), (2, 7, '2024-04-01', 0.7), "
        "(3, 8, '2024-02-01', 0.9), (4, 7, '2024-07-01', 0.8)",
        "(9, 7, '2024-09-01', 9.9)",
    ),
]


def _matrix(conn, config: dict, *, future_rows: bool) -> dict:
    with conn.transaction(force_rollback=True):
        with conn.cursor() as cur:
            for table, columns, knowable, future in TABLES:
                cur.execute(f"create temp table {table} ({columns}) on commit drop")
                cur.execute(f"insert into {table} values {knowable}")
                if future_rows:
                    cur.execute(f"insert into {table} values {future}")
            cur.execute(
                "create temp table as_of_dates (as_of_date date) on commit drop"
            )
            cur.executemany(
                "insert into as_of_dates values (%s)", [(d,) for d in AS_OF]
            )
        rows = run_featurizer(conn, config)
    return {(row["as_of_date"], row["series_id"]): row for row in rows}


@pytest.mark.parametrize(
    "config,families",
    [
        pytest.param(
            _passes_config(),
            ("PEER_", "COLOCATION_COUNT", "KDE_INTENSITY", "DEGREE", "NEIGHBOUR_"),
            id="peer-spatial-graph",
        ),
        pytest.param(_lookup_config(), (".level",), id="as-of-lookup"),
    ],
)
def test_a_row_after_the_as_of_date_moves_no_planner_pass(
    pg_conn, config, families
) -> None:
    knowable = _matrix(pg_conn, config, future_rows=False)
    with_future = _matrix(pg_conn, config, future_rows=True)

    columns = set(next(iter(knowable.values())))
    for family in families:
        assert any(family in column for column in columns), (family, sorted(columns))
    # The values are not all NULL, or "nothing moved" would prove nothing.
    assert any(
        value not in (None, 0)
        for row in knowable.values()
        for column, value in row.items()
        if any(family in column for family in families)
    )

    # The target is not cut (its rows are the cohort), so the later series row
    # is emitted in the second run. Every row both runs share must be equal.
    assert set(knowable) <= set(with_future)
    moved = [
        (key, column, value, with_future[key][column])
        for key, row in knowable.items()
        for column, value in row.items()
        if with_future[key][column] != value
    ]
    assert not moved, f"{len(moved)} cells moved, e.g. {moved[:3]}"


def test_an_interval_over_a_looked_up_value_is_cut_on_the_receiving_rows(pg_conn):
    """Issue #48. ``MAX(events.level|interval=P90D)`` is the maximum, over the
    EVENTS of the last 90 days, of the rate in force at each event. The filter
    used to read the rate's own date (``rate_ts``), a column the events never
    carried; had it existed, it would also have been the wrong window.

    Series 1, zone 7: events on 03-10 (rate 0.5 in force) and 05-20 (rate 0.7).
    As-of 06-01 both are inside 90 days; as-of 08-01 only the second is.
    """
    matrix = _matrix(pg_conn, _lookup_config(), future_rows=True)
    june = matrix[(AS_OF[0], 1)]
    august = matrix[(AS_OF[1], 1)]
    assert june["MAX(events.level)"] == 0.7
    assert june["MAX(events.level|interval=P90D)"] == 0.7
    assert june["COUNT(events.event_id|interval=P90D)"] == 2
    assert august["COUNT(events.event_id|interval=P90D)"] == 1
    assert august["MAX(events.level|interval=P90D)"] == 0.7
    assert august["MAX(events.x|interval=P90D)"] == 4.0
