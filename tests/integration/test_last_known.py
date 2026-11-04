"""``last_known`` against a hand-computed answer, on both render paths.

A small ``labs`` child, keyed ``lab_id`` and not by the parent key, so the
``MAX(labs.LAST(labs.glucose))`` recipe would fall into its trap here. The
rows carry the three cases the aggregation has to get right:

- a NULL on the latest row before the as-of date (skipped: the last *known*
  value is the previous one);
- two rows on one timestamp (resolved by the child's tiebreak, ``lab_id``,
  never by the order the rows were inserted in);
- rows dated after the as-of date (they must not move anything).

The matrix is computed with the rows inserted in two physical orders, through
the set-based pre-pass the planner takes and through the correlated subquery,
and every combination must equal the oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

from featurizer import Featurizer
from featurizer.primitives.aggregations import LastKnown

pytestmark = pytest.mark.integration

AS_OF = ("2024-03-01", "2024-06-01")

# (lab_id, patient_id, taken_at, glucose)
LABS: list[tuple[int, int, str, Optional[float]]] = [
    # Patient 1: the latest known row before 2024-06-01 is NULL, so the last
    # known value is February's 110. The July row is after both as-of dates.
    (1, 1, "2024-01-10 08:00", 100.0),
    (2, 1, "2024-02-10 08:00", 110.0),
    (3, 1, "2024-03-10 08:00", None),
    (4, 1, "2024-07-01 08:00", 999.0),
    # Patient 2: lab 11 and lab 12 tie on the timestamp. The timeline orders
    # by lab_id after the timestamp, so lab 12 is the later one: 70. Lab 13 is
    # one day after the as-of date, and lower than everything, so a leak shows.
    (10, 2, "2024-04-01 08:00", 50.0),
    (12, 2, "2024-05-20 09:00", 70.0),
    (11, 2, "2024-05-20 09:00", 60.0),
    (13, 2, "2024-06-02 08:00", 1.0),
    # Patient 3: one value, far before the as-of date: inside the whole
    # history, outside a 30-day window.
    (20, 3, "2024-01-05 08:00", 5.0),
    # Patient 4: only NULLs. Patient 5 has no labs at all.
    (30, 4, "2024-02-01 08:00", None),
    (31, 4, "2024-05-30 08:00", None),
]

WHOLE = '"LAST_KNOWN(labs.glucose)"'
WINDOW = '"LAST_KNOWN(labs.glucose|interval=P30D)"'

# (patient_id, as_of_date) -> (whole history, last 30 days), by hand.
EXPECTED: dict[tuple[int, str], tuple[Optional[float], Optional[float]]] = {
    # 2024-03-01: lab 2 (02-10) is the latest; the window 01-31..03-01 holds it.
    (1, "2024-03-01"): (110.0, 110.0),
    # 2024-06-01: lab 3 is NULL and skipped; the window 05-02..06-01 has no row.
    (1, "2024-06-01"): (110.0, None),
    (2, "2024-03-01"): (None, None),
    (2, "2024-06-01"): (70.0, 70.0),
    (3, "2024-03-01"): (5.0, None),
    (3, "2024-06-01"): (5.0, None),
    (4, "2024-03-01"): (None, None),
    (4, "2024-06-01"): (None, None),
    (5, "2024-03-01"): (None, None),
    (5, "2024-06-01"): (None, None),
}


def _config(tmp_path: Path) -> str:
    config = {
        "target": "patients",
        "max_depth": 2,
        "intervals": ["P30D"],
        "aggregations": ["last_known"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "patients",
                "table": "patients",
                "id": "patient_id",
                "variables": {},
            },
            {
                "alias": "labs",
                "table": "labs",
                "id": "lab_id",
                "temporal_ix": "taken_at",
                "variables": {"glucose": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "patients", "key": "patient_id"},
                "child": {"entity": "labs", "key": "patient_id"},
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _query(tmp_path: Path, *, correlated: bool, monkeypatch) -> str:
    if correlated:
        # Turn the ADR-0010 opt-in off for this aggregator only: the feature's
        # definition becomes the correlated subquery over ``sub``.
        monkeypatch.setattr(LastKnown, "_build_preagg", lambda self, *a, **k: None)
    query = Featurizer(_config(tmp_path)).query
    monkeypatch.undo()
    return query


def _matrix(pg_conn, query: str, rows: list) -> dict[tuple[int, str], tuple]:
    with pg_conn.cursor() as cur:
        cur.execute("drop table if exists patients, labs, as_of_dates")
        cur.execute("create temp table patients (patient_id int)")
        cur.execute("insert into patients values (1), (2), (3), (4), (5)")
        cur.execute(
            "create temp table labs (lab_id int, patient_id int, "
            "taken_at timestamp, glucose double precision)"
        )
        cur.executemany("insert into labs values (%s, %s, %s, %s)", rows)
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in AS_OF])
        cur.execute(query)
        names = [d.name for d in cur.description]
        out: dict[tuple[int, str], tuple[Any, Any]] = {}
        for row in cur.fetchall():
            record = dict(zip(names, row))
            key = (record["patient_id"], record["as_of_date"].isoformat())
            out[key] = (record[WHOLE.strip('"')], record[WINDOW.strip('"')])
    return out


@pytest.mark.parametrize("correlated", [False, True], ids=["set-based", "correlated"])
@pytest.mark.parametrize("order", ["as-listed", "reversed"])
def test_last_known_matches_the_hand_computed_answer(
    pg_conn, tmp_path, monkeypatch, correlated: bool, order: str
) -> None:
    query = _query(tmp_path, correlated=correlated, monkeypatch=monkeypatch)
    assert ("preaggs_for_patients" in query) is not correlated
    rows = LABS if order == "as-listed" else list(reversed(LABS))
    assert _matrix(pg_conn, query, rows) == EXPECTED


def test_rows_after_the_as_of_date_move_nothing(pg_conn, tmp_path, monkeypatch):
    query = _query(tmp_path, correlated=False, monkeypatch=monkeypatch)
    knowable = [row for row in LABS if row[0] not in {4, 13}]
    assert _matrix(pg_conn, query, knowable) == _matrix(pg_conn, query, LABS)


def test_set_based_and_correlated_paths_agree(pg_conn, tmp_path, monkeypatch):
    set_based = _query(tmp_path, correlated=False, monkeypatch=monkeypatch)
    correlated = _query(tmp_path, correlated=True, monkeypatch=monkeypatch)
    assert set_based != correlated
    for rows in (LABS, list(reversed(LABS))):
        assert _matrix(pg_conn, set_based, rows) == _matrix(pg_conn, correlated, rows)
