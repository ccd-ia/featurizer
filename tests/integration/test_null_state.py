"""A NULL category is a state (issue #86, ADR-0020).

``LAG`` returns NULL for two reasons: the row has no previous row, or the
previous row's value is NULL. A sequence primitive that drops ``prev IS NULL``
to skip the first row also drops every step out of a NULL state. ADR-0020's
rule: a NULL is a state for every categorical aggregation that reads a value
as a category, so a value equals the value on the same rows with each NULL
replaced by a category that appears nowhere else.

Two oracles, because one cannot see everything:

- **the sentinel sweep**: every aggregation with a categorical input (from the
  registry), minus the SQL aggregates whose definition ignores NULL inputs, is
  run on the rows with their NULLs and on the same rows with each NULL replaced
  by ``SENTINEL``. The values must be equal. A partial trigram exists with and
  without the sentinel, so this oracle cannot see it;
- **the reference sweep**: the 11 sequence primitives, on the set-based
  pre-pass and on the correlated fallback, cell by cell against the
  pure-Python reference in ``tests/_sequence_reference.py``.

The rows carry what a NULL rule can get wrong: a NULL first row, a NULL last
row, a NULL run of two, a window whose rows are all NULL, a NULL inside a tied
timestamp, one-row and two-row series, a series with no rows, and rows after
each of the two as-of dates. Every timestamp is at midnight, so the day bound
of a window and the instant bound of the read agree (issue #87 is separate).
The ``event_id``\\ s run against the insertion order and the rows are loaded
forwards and then backwards, the pattern of
``tests/integration/test_correlated_fallback_parity.py``.

A case the engine gets wrong is recorded in ``MISMATCHES`` with a strict
``xfail`` naming #86, so the list cannot be longer than what fails.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pytest

from benchmarks._db import values_equal
from featurizer import Featurizer
from featurizer.primitives.utils import get_aggregations, list_aggregations
from tests._sequence_reference import REFERENCE
from tests.test_aggregation_quoting import PREDICATES, _config

if TYPE_CHECKING:
    import psycopg

pytestmark = pytest.mark.integration

AS_OF = ("2024-04-01", "2024-06-01")
INTERVAL = "P30D"
INTERVAL_DAYS = 30
SENTINEL = "__null__"
assert SENTINEL not in PREDICATES.values()

# The SQL aggregates whose definition ignores NULL inputs: count(x),
# count(distinct x) and mode() within group. ADR-0020 keeps their SQL
# semantics; test_a_sql_aggregate_is_excluded_because_it_moves shows each one
# does move under the sentinel, so the exclusion is needed.
SQL_AGGREGATES = {
    "count": "count(x) skips NULL inputs",
    "nunique": "count(distinct x) skips NULL inputs",
    "mode": "mode() within group skips NULL inputs",
}

# (series_id, day, value), in timeline order within a series; ``_rows``
# numbers them backwards, so inside a tie the id order (the ADR-0018
# tiebreak) is the reverse of the order listed.
EVENTS: list[tuple[int, str, Optional[str]]] = [
    # 1: a NULL first row, a NULL run of two, a NULL inside the second
    # window, and a row after both as-of dates.
    (1, "2024-01-05", None),
    (1, "2024-01-20", "a"),
    (1, "2024-02-10", "b"),
    (1, "2024-03-05", None),
    (1, "2024-03-12", None),
    (1, "2024-03-20", "a"),
    (1, "2024-05-08", "b"),
    (1, "2024-05-15", None),
    (1, "2024-05-25", "a"),
    (1, "2024-07-01", "z"),
    # 2: a NULL last row at both as-of dates; a row one day after the first.
    (2, "2024-01-02", "a"),
    (2, "2024-03-05", "b"),
    (2, "2024-03-15", "a"),
    (2, "2024-03-30", None),
    (2, "2024-04-02", "z"),
    (2, "2024-05-05", "a"),
    (2, "2024-05-10", "b"),
    (2, "2024-05-31", None),
    # 3: both 30-day windows hold only NULLs; the whole history does not.
    (3, "2024-01-10", "a"),
    (3, "2024-02-01", "b"),
    (3, "2024-03-10", None),
    (3, "2024-03-25", None),
    (3, "2024-05-05", None),
    (3, "2024-05-20", None),
    # 4: a NULL inside two tied timestamps.
    (4, "2024-03-10", None),
    (4, "2024-03-10", "a"),
    (4, "2024-03-20", "b"),
    (4, "2024-05-20", "a"),
    (4, "2024-05-20", None),
    (4, "2024-05-20", "b"),
    (4, "2024-05-28", "b"),
    # 5: one row.
    (5, "2024-03-15", "a"),
    # 6: two rows (no trigram), and an empty second window.
    (6, "2024-03-01", "a"),
    (6, "2024-03-20", "b"),
    # 7: every row NULL.
    (7, "2024-02-01", None),
    (7, "2024-03-15", None),
    (7, "2024-05-15", None),
    # 9: rows after both as-of dates only. 8: no rows.
    (9, "2024-08-01", "z"),
]
SERIES = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# What the engine gets wrong on these rows, measured. Each entry is a strict
# xfail; ADR-0020's fix (#86) deletes them.
_OUT_OF_NULL = "prev IS NOT NULL drops every step out of a NULL state"
_EMPTY = (
    "; and the correlated fallback counts 0 on an empty window of a key with "
    "history, where the pre-pass gives NULL"
)
SENTINEL_MISMATCHES: dict[str, str] = {
    "markov_conditional_entropy": _OUT_OF_NULL,
    "max_transition_prob": _OUT_OF_NULL,
    "sequence_entropy": _OUT_OF_NULL,
    "state_volatility": "NULL -> x is not counted as a change",
    "transition_matrix_summary": "a (NULL, x) pair is not counted",
    "rework_count": "prev = curr: NULL -> NULL never repeats",
    "ngram_2_freq": "lag_1 IS NOT NULL drops every bigram that starts at a NULL",
    "ngram_3_freq": "lag_1 IS NOT NULL drops every trigram after a NULL",
    "time_in_current_state": "an all-NULL window has no change, so no dwell",
}
REFERENCE_MISMATCHES: dict[str, str] = {
    **{f"{name}-set_based": reason for name, reason in SENTINEL_MISMATCHES.items()},
    **{f"{name}-correlated": reason for name, reason in SENTINEL_MISMATCHES.items()},
}
REFERENCE_MISMATCHES["ngram_3_freq-set_based"] += "; a two-row series counts 1"
REFERENCE_MISMATCHES["ngram_3_freq-correlated"] += "; a two-row series counts 1"
for _name in ("state_volatility", "transition_matrix_summary", "rework_count"):
    REFERENCE_MISMATCHES[f"{_name}-correlated"] += _EMPTY


def _categorical() -> set[str]:
    return {
        name
        for name, agg in get_aggregations(list_aggregations()).items()
        if "categorical" in (getattr(agg, "input_types", None) or [])
    }


def sentinel_names() -> set[str]:
    """The aggregations the sentinel sweep covers: every categorical one but
    the SQL aggregates. ``tests/test_sweep_matrix_coverage.py`` reads it."""
    return _categorical() - set(SQL_AGGREGATES)


def _xfail(mismatches: dict[str, str], key: str) -> list[Any]:
    if key not in mismatches:
        return []
    return [pytest.mark.xfail(strict=True, reason=f"#86: {mismatches[key]}")]


def _sentinel_cases() -> Iterator[Any]:
    for name in sorted(sentinel_names()):
        yield pytest.param(name, id=name, marks=_xfail(SENTINEL_MISMATCHES, name))


def _reference_cases() -> Iterator[Any]:
    for name in sorted(REFERENCE):
        for path in ("set_based", "correlated"):
            key = f"{name}-{path}"
            yield pytest.param(
                name, path, id=key, marks=_xfail(REFERENCE_MISMATCHES, key)
            )


def _query(
    tmp_path: Path, name: str, *, correlated: bool, monkeypatch: pytest.MonkeyPatch
) -> str:
    directory = tmp_path / ("correlated" if correlated else "set_based")
    directory.mkdir()
    config = _config(directory, "x", [name], "categorical", intervals=(INTERVAL,))
    if not correlated:
        return Featurizer(config, validate=False).query
    cls = type(get_aggregations([name])[name])
    with monkeypatch.context() as patch:
        patch.setattr(cls, "_build_preagg", lambda self, *a, **k: None)
        return Featurizer(config, validate=False).query


def _rows(*, sentinel: bool) -> list[tuple[int, int, str, Optional[str]]]:
    total = len(EVENTS)
    return [
        (
            total - i,
            series,
            day,
            SENTINEL if sentinel and value is None else value,
        )
        for i, (series, day, value) in enumerate(EVENTS)
    ]


def _seed(cur: psycopg.Cursor[Any]) -> None:
    cur.execute("create temp table series (series_id int)")
    cur.executemany("insert into pg_temp.series values (%s)", [(s,) for s in SERIES])
    cur.execute(
        "create temp table events (event_id int, series_id int, ts timestamp, x text)"
    )
    cur.execute("create temp table as_of_dates (as_of_date date)")
    cur.executemany(
        "insert into pg_temp.as_of_dates values (%s)", [(d,) for d in AS_OF]
    )


def _load(cur: psycopg.Cursor[Any], rows: list[Any]) -> None:
    cur.execute("truncate pg_temp.events")
    cur.executemany("insert into pg_temp.events values (%s, %s, %s, %s)", rows)


def _run(cur: psycopg.Cursor[Any], query: str) -> list[dict[str, Any]]:
    cur.execute(query)
    assert cur.description is not None
    names = [d.name for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _features(row: dict[str, Any], name: str) -> list[str]:
    return [c for c in row if c.upper().startswith(name.upper() + "(")]


def _assert_not_silent(rows: list[dict[str, Any]], features: list[str]) -> None:
    silent = [c for c in features if all(row[c] is None for row in rows)]
    assert not silent, f"{silent} NULL everywhere: the comparison proves nothing"


def _loads() -> Iterator[tuple[str, bool]]:
    for order in ("forwards", "backwards"):
        yield order, order == "backwards"


# ------------------------------------------------------------ the sentinel


def _sentinel_diff(
    pg_conn: psycopg.Connection[Any], query: str, name: str
) -> tuple[bool, str]:
    """Run ``query`` on the rows with NULLs and with the sentinel, both loads."""
    with pg_conn.cursor() as cur:
        _seed(cur)
        for order, reverse in _loads():
            results = []
            for sentinel in (False, True):
                rows = _rows(sentinel=sentinel)
                _load(cur, rows[::-1] if reverse else rows)
                results.append(_run(cur, query))
            with_nulls, with_sentinel = results
            assert len(with_nulls) == len(SERIES) * len(AS_OF), order
            features = _features(with_nulls[0], name)
            assert features, f"{name} emitted no feature column"
            _assert_not_silent(with_nulls, features)
            # Align on the target key first: values_equal orders rows by every
            # column, the feature columns first, so two rows that differ in a
            # value would be compared with the wrong partners.
            wrong = [
                f"series {e['series_id']} @ {_day(e['as_of_date'])}: {reason}"
                for e, s in zip(_by_key(with_nulls), _by_key(with_sentinel))
                for equal, reason in [values_equal([e], [s])]
                if not equal
            ]
            if wrong:
                return False, f"{order}:\n" + "\n".join(wrong)
    return True, "equal"


def _by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["series_id"], _day(r["as_of_date"])))


@pytest.mark.parametrize("name", list(_sentinel_cases()))
def test_a_null_category_gives_the_value_of_an_unused_category(
    pg_conn: psycopg.Connection[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    query = _query(tmp_path, name, correlated=False, monkeypatch=monkeypatch)
    equal, reason = _sentinel_diff(pg_conn, query, name)
    assert equal, f"{name}: a NULL is not a state: {reason}"


@pytest.mark.parametrize("name", sorted(SQL_AGGREGATES))
def test_a_sql_aggregate_is_excluded_because_it_moves(
    pg_conn: psycopg.Connection[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """An exclusion that changes nothing is not needed: drop it from
    SQL_AGGREGATES and let the sentinel sweep cover the aggregation."""
    query = _query(tmp_path, name, correlated=False, monkeypatch=monkeypatch)
    equal, _ = _sentinel_diff(pg_conn, query, name)
    assert not equal, f"{name} gives equal values under the sentinel"


# ----------------------------------------------------------- the reference


def _day(value: Any) -> date:
    return value.date() if isinstance(value, datetime) else value


def _window(
    series: int, as_of: date, interval: bool
) -> list[tuple[date, Optional[str]]]:
    """The rows of ``series`` the read and the window let through, in the
    engine's timeline order: the day, then the event id (ADR-0018)."""
    start = as_of - timedelta(days=INTERVAL_DAYS)
    kept = [
        (date.fromisoformat(day), event_id, value)
        for event_id, s, day, value in _rows(sentinel=False)
        if s == series
        and date.fromisoformat(day) <= as_of
        and (not interval or date.fromisoformat(day) >= start)
    ]
    return [(day, value) for day, _, value in sorted(kept, key=lambda r: r[:2])]


def _same(engine: Any, expected: Any) -> bool:
    if engine is None or expected is None:
        return engine is None and expected is None
    if isinstance(engine, (float, Decimal)) or isinstance(expected, float):
        return math.isclose(float(engine), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return engine == expected


@pytest.mark.parametrize(("name", "path"), list(_reference_cases()))
def test_a_sequence_primitive_gives_the_reference_value(
    pg_conn: psycopg.Connection[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    path: str,
) -> None:
    correlated = path == "correlated"
    query = _query(tmp_path, name, correlated=correlated, monkeypatch=monkeypatch)
    assert ("preaggs_for_series" in query) is not correlated, path
    with pg_conn.cursor() as cur:
        _seed(cur)
        for order, reverse in _loads():
            rows = _rows(sentinel=False)
            _load(cur, rows[::-1] if reverse else rows)
            result = _run(cur, query)
            assert len(result) == len(SERIES) * len(AS_OF), order
            features = _features(result[0], name)
            assert len(features) == 2, f"{name}: expected two columns, {features}"
            _assert_not_silent(result, features)
            wrong = []
            for row in result:
                as_of = _day(row["as_of_date"])
                for column in features:
                    interval = "interval=" in column
                    window = _window(row["series_id"], as_of, interval)
                    expected = REFERENCE[name](window, as_of)
                    if not _same(row[column], expected):
                        wrong.append(
                            f"series {row['series_id']} @ {as_of} {column}: "
                            f"engine {row[column]!r}, reference {expected!r}, "
                            f"rows {[v for _, v in window]}"
                        )
            assert not wrong, f"{name} ({path}, {order}):\n" + "\n".join(wrong)
