"""Live-PG proof of the native 1-hop pass's double causal bound.

The seeded data plants BOTH leakage vectors: a future edge (must not appear in
degree or bring its neighbour's state in) and a future neighbour-state row
(must not enter the mean/share even though its edge is knowable). Expected
values are hand-computed under the pass's documented incidence semantics —
each knowable (edge, state) pair weighs once.
"""

from __future__ import annotations

from datetime import date

import pytest

from ._harness import create_temp_table, run_featurizer
from ._realistic import feature, make_as_of_dates

pytestmark = [pytest.mark.integration]

AS_OF_DATES = ["2020-06-01", "2020-12-31"]


def _seed(conn) -> None:
    create_temp_table(conn, "facilities", [("facility_id", "int")], [(1,), (2,), (3,)])
    create_temp_table(
        conn,
        "contact_edges",
        [("src_id", "int"), ("dst_id", "int"), ("contacted_at", "date")],
        [
            (1, 2, date(2020, 1, 15)),
            (1, 3, date(2020, 11, 1)),  # planted FUTURE edge at the June cut
            (2, 3, date(2020, 2, 1)),
        ],
    )
    create_temp_table(
        conn,
        "facility_states",
        [
            ("facility_id", "int"),
            ("valid_at", "date"),
            ("risk_score", "double precision"),
            ("flagged", "boolean"),
        ],
        [
            (2, date(2020, 1, 1), 10.0, True),
            (3, date(2020, 3, 1), 100.0, False),
            (2, date(2020, 9, 15), 999.0, False),  # planted FUTURE state
        ],
    )


CONFIG = {
    "target": "facilities",
    "max_depth": 1,
    "intervals": ["P3M"],
    "aggregations": ["count"],
    "transformations": ["identity"],
    "entities": [
        {"alias": "facilities", "table": "facilities", "id": "facility_id"},
        {
            "alias": "states",
            "table": "facility_states",
            "id": "facility_id",
            "temporal_ix": "valid_at",
            "variables": {
                "risk_score": {"type": "numeric"},
                "flagged": {"type": "boolean"},
            },
        },
    ],
    "graph_relationships": [
        {
            "name": "contacts",
            "left": "facilities",
            "right": "states",
            "edges": {
                "table": "contact_edges",
                "source": "src_id",
                "target": "dst_id",
                "timestamp": "contacted_at",
            },
            "directed": True,
        }
    ],
}


def _run(conn):
    _seed(conn)
    make_as_of_dates(conn, AS_OF_DATES)
    return run_featurizer(conn, CONFIG)


def _cell(rows, as_of, entity_id, col):
    return feature(
        rows, as_of=as_of, id_col="facility_id", entity_id=entity_id, col_substr=col
    )


def test_planted_future_edge_is_excluded_from_degree(pg_conn):
    rows = _run(pg_conn)
    # June: only the January edge is knowable; November's must not count.
    assert int(_cell(rows, "2020-06-01", 1, "DEGREE(contacts)")) == 1
    assert int(_cell(rows, "2020-12-31", 1, "DEGREE(contacts)")) == 2


def test_windowed_degree_counts_only_the_interval(pg_conn):
    rows = _run(pg_conn)
    win = "DEGREE(contacts|interval=P3M)"
    # June window [Mar 1, Jun 1]: the January edge is knowable but outside.
    assert int(_cell(rows, "2020-06-01", 1, win)) == 0
    # December window [Sep 30, Dec 31] holds exactly the November edge.
    assert int(_cell(rows, "2020-12-31", 1, win)) == 1


def test_planted_future_neighbour_state_is_excluded(pg_conn):
    rows = _run(pg_conn)
    mean = "NEIGHBOUR_MEAN(contacts.risk_score)"
    share = "NEIGHBOUR_SHARE(contacts.flagged)"
    # June, facility 1: neighbour 2 only, and only its January state row
    # (risk 10, flagged) — the September 999-risk row must not leak back.
    assert float(_cell(rows, "2020-06-01", 1, mean)) == 10.0
    assert float(_cell(rows, "2020-06-01", 1, share)) == 1.0
    # December, facility 1: incidence semantics — edge→2 matches two knowable
    # state rows (10, 999), edge→3 one (100): mean of three, one flagged.
    assert float(_cell(rows, "2020-12-31", 1, mean)) == pytest.approx(
        (10.0 + 999.0 + 100.0) / 3
    )
    assert float(_cell(rows, "2020-12-31", 1, share)) == pytest.approx(1 / 3)


def test_directed_degree_and_isolated_node(pg_conn):
    rows = _run(pg_conn)
    # Directed: facility 2 has one out-edge (to 3); its state is knowable at
    # June (March row), so the neighbour mean is 100.
    assert int(_cell(rows, "2020-06-01", 2, "DEGREE(contacts)")) == 1
    assert (
        float(_cell(rows, "2020-06-01", 2, "NEIGHBOUR_MEAN(contacts.risk_score)"))
        == 100.0
    )
    # Facility 3 has no out-edges: NULL features, never fabricated zeros.
    assert _cell(rows, "2020-12-31", 3, "DEGREE(contacts)") is None
