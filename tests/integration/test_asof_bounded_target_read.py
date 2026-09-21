"""A target with a temporal index is read as of the as-of date too (issue #49).

Issue #27 cut every NON-target read on the as-of date and left the target alone
on purpose: its rows are the cohort. For a target that is one row per entity
that is right, and such a target has no temporal index to cut on. For an
event-like target — a temporal index, several rows per id — it left the leak
#27 closed for children. Measured on 8e1222e, one later visit added:

- the row dated 2024-09-01 was EMITTED under the as-of date 2024-06-01;
- it moved ``cross_entity_zscore`` on 5 of the 5 knowable rows, ``cusum`` and
  ``last`` on 3 of 5, ``percent_rank`` on 2 of 5.

A row that does not exist yet at a date is not part of that date's matrix. The
target's read now carries the same cut as everybody else's (ADR-0017).
"""

from __future__ import annotations

import datetime

import pytest

from featurizer.primitives.utils import get_transformers, list_transformations

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration

AS_OF = [datetime.date(2024, 6, 1), datetime.date(2024, 10, 1)]
KNOWABLE = [
    (1, "2024-01-10", 10.0),
    (1, "2024-03-10", 30.0),
    (1, "2024-05-10", 20.0),
    (2, "2024-02-01", 5.0),
    (2, "2024-04-01", 50.0),
]
# After the first as-of date, before the second. Lower than every knowable cost,
# so it moves every rank and mean it is allowed to touch.
LATER = (1, "2024-09-01", 0.5)
NOT_STANDALONE = {"identity", "in_array"}


def _config(transformer: str) -> dict:
    return {
        "target": "visits",
        "max_depth": 1,
        "intervals": [],
        "aggregations": [],
        "transformations": ["identity", transformer],
        "entities": [
            {
                "alias": "visits",
                "table": "visits",
                "id": "patient_id",
                "temporal_ix": "visited",
                "variables": {"cost": {"type": "numeric"}},
            }
        ],
    }


def _matrix(conn, config: dict, *, later_row: bool) -> dict:
    with conn.transaction(force_rollback=True):
        rows = KNOWABLE + ([LATER] if later_row else [])
        create_temp_table(
            conn,
            "visits",
            [("patient_id", "int"), ("visited", "date"), ("cost", "double precision")],
            rows,
        )
        create_temp_table(
            conn, "as_of_dates", [("as_of_date", "date")], [(d,) for d in AS_OF]
        )
        out = run_featurizer(conn, config)
    # ``visited`` is projected only while it is the temporal index; ``cost`` is
    # unique per row and stands in for it otherwise.
    return {
        (r["as_of_date"], r["patient_id"], r.get("visited", r["cost"])): r for r in out
    }


def _numeric_transformers():
    for name in sorted(list_transformations()):
        if name in NOT_STANDALONE:
            continue
        types = getattr(get_transformers([name])[name], "input_types", None) or [
            "numeric"
        ]
        if "numeric" in types:
            yield name


@pytest.mark.parametrize("transformer", list(_numeric_transformers()))
def test_a_later_target_row_moves_nothing_under_an_earlier_date(
    pg_conn, transformer
) -> None:
    config = _config(transformer)
    without = _matrix(pg_conn, config, later_row=False)
    with_later = _matrix(pg_conn, config, later_row=True)
    first = AS_OF[0]
    before = {k: v for k, v in without.items() if k[0] == first}
    after = {k: v for k, v in with_later.items() if k[0] == first}
    assert after == before


def test_a_row_is_emitted_from_the_first_as_of_date_it_exists_at(pg_conn) -> None:
    matrix = _matrix(pg_conn, _config("abs"), later_row=True)
    later = datetime.date(2024, 9, 1)
    assert (AS_OF[0], 1, later) not in matrix
    assert (AS_OF[1], 1, later) in matrix
    # The five knowable rows are there under both dates.
    assert sum(1 for key in matrix if key[0] == AS_OF[0]) == 5
    assert sum(1 for key in matrix if key[0] == AS_OF[1]) == 6


def test_a_target_without_a_temporal_index_returns_every_row(pg_conn) -> None:
    """One row per entity, no timeline: nothing to cut on, nothing changes.
    Every triage-pg config is this shape."""
    config = _config("abs")
    del config["entities"][0]["temporal_ix"]
    config["transformations"] = ["identity", "abs"]
    matrix = _matrix(pg_conn, config, later_row=True)
    assert sum(1 for key in matrix if key[0] == AS_OF[0]) == 6


def test_the_exclusive_boundary_drops_a_row_dated_on_the_as_of_date(pg_conn) -> None:
    config = {**_config("abs"), "as_of_boundary": "exclusive"}
    with pg_conn.transaction(force_rollback=True):
        create_temp_table(
            pg_conn,
            "visits",
            [("patient_id", "int"), ("visited", "date"), ("cost", "double precision")],
            KNOWABLE + [(1, AS_OF[0].isoformat(), 7.0)],
        )
        create_temp_table(
            pg_conn, "as_of_dates", [("as_of_date", "date")], [(AS_OF[0],)]
        )
        rows = run_featurizer(pg_conn, config)
    assert len(rows) == 5
