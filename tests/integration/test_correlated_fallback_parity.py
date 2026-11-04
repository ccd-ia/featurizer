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
  both paths;
- a NULL input;
- a row after each as-of date, with a value that would show if it leaked;
- two as-of dates, and a 30-day interval beside the whole history.

A mismatch is a finding to file, not to fix here: it is recorded in
``MISMATCHES`` with a strict ``xfail`` that names the aggregator.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from featurizer import Featurizer
from featurizer.primitives.aggregations import SubqueryAggregator
from featurizer.primitives.utils import get_aggregations, list_aggregations

pytestmark = pytest.mark.integration

AS_OF = ("2024-04-01", "2024-06-01")
SQL_TYPE = {"numeric": "double precision", "categorical": "text"}

# (series_id, ts, numeric x, categorical x). Rows are given in an order that is
# not the timeline's, so neither path is helped by the insertion order.
EVENTS: list[tuple[int, str, float | None, str | None]] = [
    # Series 1: two ties, one between an ``a`` and a ``b``; a NULL; and a row
    # after both as-of dates that would move every statistic.
    (1, "2024-01-05", 1.5, "a"),
    (1, "2024-01-20", 2.0, "a"),
    (1, "2024-01-20", 6.5, "b"),
    (1, "2024-02-14", None, None),
    (1, "2024-03-03", 4.0, "b"),
    (1, "2024-03-03", 0.5, "c"),
    (1, "2024-03-20", 3.0, "a"),
    (1, "2024-05-10", 8.0, "c"),
    (1, "2024-05-25", 2.5, "a"),
    (1, "2024-05-25", 7.0, "b"),
    (1, "2024-07-01", 999.0, "z"),
    # Series 2: dense in the last 30 days of both as-of dates, a tie on the
    # latest row, and a row one day after the first as-of date.
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


def _cases():
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


def _input_type(name: str) -> str:
    """The column the config declares: an ``index`` aggregation reads the
    temporal index, and is given a numeric variable beside it."""
    (input_type,) = get_aggregations([name])[name].input_types
    return "numeric" if input_type == "index" else input_type


def _config(tmp_path: Path, name: str) -> str:
    config = {
        "target": "series",
        "max_depth": 2,
        "intervals": ["P30D"],
        "aggregations": [name],
        "transformations": ["identity"],
        "entities": [
            {"alias": "series", "table": "series", "id": "series_id"},
            {
                "alias": "events",
                "table": "events",
                "id": "event_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": _input_type(name)}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_id"},
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _query(tmp_path: Path, name: str, *, correlated: bool, monkeypatch) -> str:
    if correlated:
        # Turn the ADR-0010 opt-in off for this aggregator's class only: the
        # feature's definition becomes the correlated subquery over ``sub``.
        cls = type(get_aggregations([name])[name])
        monkeypatch.setattr(cls, "_build_preagg", lambda self, *a, **k: None)
    try:
        return Featurizer(_config(tmp_path, name), validate=False).query
    finally:
        monkeypatch.undo()


def _matrix(pg_conn, query: str, vtype: str) -> dict[tuple[int, str], dict]:
    position = 2 if vtype == "numeric" else 3
    rows = [
        (event_id, row[0], row[1], row[position])
        for event_id, row in enumerate(EVENTS, start=1)
    ]
    with pg_conn.cursor() as cur:
        cur.execute("drop table if exists series, events, as_of_dates")
        cur.execute("create temp table series (series_id int)")
        cur.executemany("insert into series values (%s)", [(s,) for s in SERIES])
        cur.execute(
            "create temp table events (event_id int, series_id int, "
            f"ts timestamp, x {SQL_TYPE[vtype]})"
        )
        cur.executemany("insert into events values (%s, %s, %s, %s)", rows)
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in AS_OF])
        cur.execute(query)
        names = [d.name for d in cur.description]
        out: dict[tuple[int, str], dict[str, Any]] = {}
        for row in cur.fetchall():
            record = dict(zip(names, row))
            out[(record["series_id"], record["as_of_date"].isoformat())] = record
    return out


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


@pytest.mark.parametrize("name", list(_cases()))
def test_the_correlated_fallback_gives_the_prepass_values(
    pg_conn, tmp_path, monkeypatch, name: str
) -> None:
    set_based = _query(tmp_path, name, correlated=False, monkeypatch=monkeypatch)
    correlated = _query(tmp_path, name, correlated=True, monkeypatch=monkeypatch)
    # Both paths are what the test says they are.
    assert "preaggs_for_series" in set_based
    assert "preaggs_for_series" not in correlated

    vtype = _input_type(name)
    expected = _matrix(pg_conn, set_based, vtype)
    actual = _matrix(pg_conn, correlated, vtype)

    assert expected, "the query returned nothing to compare"
    assert set(actual) == set(expected)
    features = [
        column
        for column in next(iter(expected.values()))
        if column.upper().startswith(name.upper() + "(")
    ]
    # A whole-history and a P30D column; the drift pair compares two windows
    # and so emits the interval column only.
    assert features, f"{name} emitted no feature to compare"
    assert any(row[c] is not None for row in expected.values() for c in features), (
        f"{name} is NULL everywhere: the comparison would prove nothing"
    )
    moved = sorted(
        f"{column} @ {key}: pre-pass {expected[key][column]!r}, "
        f"correlated {actual[key][column]!r}"
        for key in expected
        for column in expected[key]
        if not _same(expected[key][column], actual[key][column])
    )
    assert moved == [], f"{name}: the correlated fallback differs:\n" + "\n".join(moved)
