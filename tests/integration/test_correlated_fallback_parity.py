"""The correlated fallback computes what the set-based pre-pass computes (issue #83).

Every ``SubqueryAggregator`` with an ADR-0010 pre-pass keeps a correlated
subquery as its fallback, and the planner never renders it. #83 was a fallback
that could not even be built. This sweep goes one step further than the
DB-free one in ``tests/primitives/test_correlated_fallback.py``: for each such
aggregator (from the registry), the same config is rendered twice, once as the
planner renders it and once with that class's ``_build_preagg`` turned off (the
pattern of ``tests/integration/test_last_known.py``), and both queries run on
the same rows. The values must be equal.

The rows carry what a fallback most easily gets wrong:

- tied timestamps, the tied rows differing in the input column, so the order
  of the timeline matters (ADR-0018) and must come from the same tiebreak on
  both paths. The ``event_id``\\ s run against the insertion order, and the rows
  are loaded forwards and then backwards, so a path that orders by the
  timestamp alone reads a tie in the physical order on one of the two loads
  and the tiebreak order on the other;
- a NULL input;
- rows on an as-of date, at midnight and at midday, where an inclusive or
  exclusive upper bound decides what is read;
- a row after each as-of date, with a value that would show if it leaked;
- two as-of dates, and a 30-day interval beside the whole history.

A mismatch is a finding to file, not to fix here: it is recorded in
``MISMATCHES`` with a strict ``xfail`` that names the aggregator.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from benchmarks._db import values_equal
from featurizer import Featurizer
from featurizer.primitives.aggregations import (
    SubqueryAggregator,
    TwoWindowDriftAggregator,
)
from featurizer.primitives.utils import get_aggregations, list_aggregations
from tests.test_aggregation_quoting import SQL_TYPE, _config, _input_type

if TYPE_CHECKING:
    import psycopg

pytestmark = pytest.mark.integration

AS_OF = ("2024-04-01", "2024-06-01")
INTERVAL = "P30D"

# (series_id, ts, numeric x, categorical x), in timeline order within a series.
# ``_rows`` numbers them backwards, so inside a tie the id order (the ADR-0018
# tiebreak) is the reverse of the order the rows are listed in.
EVENTS: list[tuple[int, str, float | None, str | None]] = [
    # Series 1: two ties, one between an ``a`` and a ``b``; a NULL; a row at
    # midnight on the first as-of date; and a row after both as-of dates that
    # would move every statistic.
    (1, "2024-01-05", 1.5, "a"),
    (1, "2024-01-20", 2.0, "a"),
    (1, "2024-01-20", 6.5, "b"),
    (1, "2024-02-14", None, None),
    (1, "2024-03-03", 4.0, "b"),
    (1, "2024-03-03", 0.5, "c"),
    (1, "2024-03-20", 3.0, "a"),
    (1, "2024-04-01", 3.25, "c"),
    (1, "2024-05-10", 8.0, "c"),
    (1, "2024-05-25", 2.5, "a"),
    (1, "2024-05-25", 7.0, "b"),
    (1, "2024-07-01", 999.0, "z"),
    # Series 2: dense in the last 30 days of both as-of dates, a tie on the
    # latest row, a row at midday on the second as-of date, and a row one day
    # after the first as-of date.
    (2, "2024-01-02", 5.0, "b"),
    (2, "2024-03-05", 1.0, "b"),
    (2, "2024-03-15", 2.0, "c"),
    (2, "2024-03-25", 3.5, "b"),
    (2, "2024-03-31", 4.5, "a"),
    (2, "2024-03-31", 0.25, "c"),
    (2, "2024-04-02", 500.0, "z"),
    (2, "2024-05-15", 6.0, "a"),
    (2, "2024-05-30", None, "a"),
    (2, "2024-05-30", 9.0, None),
    (2, "2024-06-01 12:00", 5.5, "b"),
    # Series 3: a constant categorical, numbers with a NULL in the middle.
    (3, "2024-02-01", 2.0, "a"),
    (3, "2024-02-11", None, "a"),
    (3, "2024-02-21", 2.0, "a"),
    (3, "2024-03-02", 4.0, "a"),
    (3, "2024-03-02", 8.0, "a"),
    (3, "2024-05-20", 16.0, "a"),
    # Series 4: one row, and only after both as-of dates. Series 5: no rows.
    (4, "2024-08-01", 42.0, "z"),
]
SERIES = (1, 2, 3, 4, 5)

# Aggregators whose correlated fallback does not agree with their pre-pass on
# these rows: the name, and what was measured. Each is a finding to file.
MISMATCHES: dict[str, str] = {}


def _has_prepass(agg: SubqueryAggregator) -> bool:
    return type(agg)._build_preagg is not SubqueryAggregator._build_preagg


def _cases() -> Iterator[Any]:
    aggregations = get_aggregations(list_aggregations())
    for name, agg in sorted(aggregations.items()):
        if not isinstance(agg, SubqueryAggregator) or not _has_prepass(agg):
            continue
        marks = (
            [pytest.mark.xfail(strict=True, reason=f"{name}: {MISMATCHES[name]}")]
            if name in MISMATCHES
            else []
        )
        yield pytest.param(name, id=name, marks=marks)


def _vtype(name: str) -> str:
    """The type of ``x``: an ``index`` aggregation reads the temporal index,
    and is given a numeric variable beside it."""
    input_type = _input_type(name)
    return "numeric" if input_type == "index" else input_type


def _query(
    tmp_path: Path, name: str, *, correlated: bool, monkeypatch: pytest.MonkeyPatch
) -> str:
    directory = tmp_path / ("correlated" if correlated else "set_based")
    directory.mkdir()
    config = _config(directory, "x", [name], _vtype(name), intervals=(INTERVAL,))
    if not correlated:
        return Featurizer(config, validate=False).query
    # Turn the ADR-0010 opt-in off for this aggregator's class only: the
    # feature's definition becomes the correlated subquery over ``sub``.
    cls = type(get_aggregations([name])[name])
    with monkeypatch.context() as patch:
        patch.setattr(cls, "_build_preagg", lambda self, *a, **k: None)
        return Featurizer(config, validate=False).query


def _rows(vtype: str) -> list[tuple[int, int, str, Any]]:
    position = 2 if vtype == "numeric" else 3
    total = len(EVENTS)
    return [(total - i, row[0], row[1], row[position]) for i, row in enumerate(EVENTS)]


def _seed(cur: psycopg.Cursor[Any], vtype: str) -> None:
    """Create the three TEMP tables once; the loads fill ``events``."""
    cur.execute("create temp table series (series_id int)")
    cur.executemany("insert into pg_temp.series values (%s)", [(s,) for s in SERIES])
    cur.execute(
        "create temp table events (event_id int, series_id int, "
        f"ts timestamp, x {SQL_TYPE[vtype]})"
    )
    cur.execute("create temp table as_of_dates (as_of_date date)")
    cur.executemany(
        "insert into pg_temp.as_of_dates values (%s)", [(d,) for d in AS_OF]
    )


def _load(cur: psycopg.Cursor[Any], rows: list[tuple[int, int, str, Any]]) -> None:
    cur.execute("truncate pg_temp.events")
    cur.executemany("insert into pg_temp.events values (%s, %s, %s, %s)", rows)


def _run(cur: psycopg.Cursor[Any], query: str) -> list[dict[str, Any]]:
    cur.execute(query)
    assert cur.description is not None
    names = [d.name for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


@pytest.mark.parametrize("name", list(_cases()))
def test_the_correlated_fallback_gives_the_prepass_values(
    pg_conn: psycopg.Connection[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    set_based = _query(tmp_path, name, correlated=False, monkeypatch=monkeypatch)
    correlated = _query(tmp_path, name, correlated=True, monkeypatch=monkeypatch)
    # Both paths are what the test says they are.
    assert "preaggs_for_series" in set_based
    assert "preaggs_for_series" not in correlated

    vtype = _vtype(name)
    rows = _rows(vtype)
    with pg_conn.cursor() as cur:
        _seed(cur, vtype)
        for order, load in (("forwards", rows), ("backwards", rows[::-1])):
            _load(cur, load)
            expected = _run(cur, set_based)
            actual = _run(cur, correlated)

            assert len(expected) == len(SERIES) * len(AS_OF), (
                f"{order}: the set-based query returned {len(expected)} rows"
            )
            features = [
                column
                for column in expected[0]
                if column.upper().startswith(name.upper() + "(")
            ]
            # A whole-history and a P30D column; the drift pair compares two
            # windows and so emits the interval column only.
            agg = get_aggregations([name])[name]
            assert any(f"interval={INTERVAL}" in c for c in features), (
                f"{name} emitted no {INTERVAL} column"
            )
            assert isinstance(agg, TwoWindowDriftAggregator) or any(
                "interval=" not in c for c in features
            ), f"{name} emitted no whole-history column"
            silent = [c for c in features if all(row[c] is None for row in expected)]
            assert not silent, (
                f"{name}: {silent} NULL everywhere, so the comparison would "
                "prove nothing about them"
            )

            equal, reason = values_equal(expected, actual)
            assert equal, f"{name} ({order}): the correlated fallback differs: {reason}"
