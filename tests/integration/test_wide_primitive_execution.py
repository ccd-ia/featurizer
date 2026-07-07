"""Execute the wide-primitive SQL fixes against real PostgreSQL.

The four bugs found by stress-testing against triage-pg's live datasets were all
dialect defects that string-shape tests could not catch — they only surface when
the generated SQL runs. This module executes the fixed temporal aggregators
(``event_rate``, ``time_span``, the ``gap_*`` family, ``burstiness``) and
``geometric_mean`` over BOTH a ``date`` and a ``timestamp`` temporal column, with
hand-computed expected values, so a regression re-breaks here immediately.

Fixture: one ``ego`` (entity 1) with three child events. Amounts 1, 4, 16 →
geometric mean ``(1·4·16)^(1/3) = 4``. Date gaps 2 and 3 days → gap_mean 2.5,
span 5 days, event_rate 3/5 = 0.6. The timestamp fixture uses 0.5- and 1.0-day
gaps (gap_mean 0.75) to prove fractional-day resolution and that
``STDDEV(interval)`` no longer errors.
"""

from __future__ import annotations

import math

import pytest

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration

_TEMPORAL_AGGS = [
    "event_rate",
    "time_span",
    "gap_mean",
    "gap_stddev",
    "gap_min",
    "gap_max",
    "gap_cv",
    "burstiness",
]


def _config() -> dict:
    return {
        "target": "ego",
        "max_depth": 2,
        "intervals": [],
        "aggregations": _TEMPORAL_AGGS + ["geometric_mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "ego", "table": "ego", "id": "eid", "variables": {}},
            {
                "alias": "evt",
                "table": "evt",
                "id": None,
                "temporal_ix": "ts",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "ego", "key": "eid"},
                "child": {"entity": "evt", "key": "eid"},
            },
        ],
    }


def _seed(conn, ts_type: str, ts_values: list[str]) -> None:
    create_temp_table(conn, "ego", [("eid", "int")], [(1,)])
    create_temp_table(
        conn,
        "evt",
        [("eid", "int"), ("ts", ts_type), ("amount", "numeric")],
        [(1, ts_values[0], 1.0), (1, ts_values[1], 4.0), (1, ts_values[2], 16.0)],
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])


def _run(conn, ts_type: str, ts_values: list[str]) -> dict:
    _seed(conn, ts_type, ts_values)
    rows = run_featurizer(conn, _config())
    assert len(rows) == 1  # one ego × one as_of_date
    return rows[0]


def test_temporal_primitives_execute_on_date_column(pg_conn):
    row = _run(pg_conn, "date", ["2023-01-01", "2023-01-03", "2023-01-06"])
    # geometric mean of 1, 4, 16
    assert float(row["GEOMETRIC_MEAN(evt.amount)"]) == pytest.approx(4.0)
    # span 5 days, 3 events → 0.6/day; gaps [2, 3] days
    assert float(row["TIME_SPAN(evt.ts)"]) == pytest.approx(5.0)
    assert float(row["EVENT_RATE(evt.ts)"]) == pytest.approx(0.6)
    assert float(row["GAP_MEAN(evt.ts)"]) == pytest.approx(2.5)
    assert float(row["GAP_MIN(evt.ts)"]) == pytest.approx(2.0)
    assert float(row["GAP_MAX(evt.ts)"]) == pytest.approx(3.0)
    # sample stddev of [2, 3] = sqrt(0.5); every value finite (no interval error)
    assert float(row["GAP_STDDEV(evt.ts)"]) == pytest.approx(math.sqrt(0.5))
    assert row["GAP_CV(evt.ts)"] is not None
    assert row["BURSTINESS(evt.ts)"] is not None


def test_temporal_primitives_execute_on_timestamp_column(pg_conn):
    # Sub-day gaps: 0.5 day then 1.0 day → mean 0.75 (fractional days).
    row = _run(
        pg_conn,
        "timestamp",
        ["2023-01-01 00:00", "2023-01-01 12:00", "2023-01-02 12:00"],
    )
    assert float(row["GEOMETRIC_MEAN(evt.amount)"]) == pytest.approx(4.0)
    assert float(row["GAP_MEAN(evt.ts)"]) == pytest.approx(0.75)
    assert float(row["GAP_MIN(evt.ts)"]) == pytest.approx(0.5)
    assert float(row["GAP_MAX(evt.ts)"]) == pytest.approx(1.0)
    # The bug this guards: STDDEV over a timestamp gap used to raise
    # "function stddev(interval) does not exist".
    assert row["GAP_STDDEV(evt.ts)"] is not None
    assert row["BURSTINESS(evt.ts)"] is not None


def _stat_config() -> dict:
    return {
        "target": "ego",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["skewness", "kurtosis"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "ego", "table": "ego", "id": "eid", "variables": {}},
            {
                "alias": "evt",
                "table": "evt",
                "id": None,
                "temporal_ix": "ts",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "ego", "key": "eid"},
                "child": {"entity": "evt", "key": "eid"},
            },
        ],
    }


def test_skewness_kurtosis_execute_with_correct_moments(pg_conn):
    # Symmetric series 1..5: population skewness 0, population kurtosis
    # m4/m2^2 = 6.8/4 = 1.7. The old ``**`` / bare-column form did not execute.
    create_temp_table(pg_conn, "ego", [("eid", "int")], [(1,)])
    create_temp_table(
        pg_conn,
        "evt",
        [("eid", "int"), ("ts", "date"), ("amount", "numeric")],
        [(1, f"2023-01-0{i}", float(i)) for i in range(1, 6)],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )
    row = run_featurizer(pg_conn, _stat_config())[0]
    assert float(row["SKEWNESS(evt.amount)"]) == pytest.approx(0.0, abs=1e-9)
    assert float(row["KURTOSIS(evt.amount)"]) == pytest.approx(1.7)


def test_geometric_mean_is_null_for_non_positive(pg_conn):
    # min(amount) <= 0 → NULL (undefined domain), not a crash or wrong number.
    create_temp_table(pg_conn, "ego", [("eid", "int")], [(1,)])
    create_temp_table(
        pg_conn,
        "evt",
        [("eid", "int"), ("ts", "date"), ("amount", "numeric")],
        [(1, "2023-01-01", 4.0), (1, "2023-01-02", -1.0)],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )
    rows = run_featurizer(pg_conn, _config())
    assert rows[0]["GEOMETRIC_MEAN(evt.amount)"] is None
