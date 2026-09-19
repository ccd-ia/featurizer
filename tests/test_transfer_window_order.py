# coding: utf-8

"""A window over a transferred feature walks the target's timeline (issue #21).

A window transformer is assembled from two halves that came from different
places: the PARTITION BY was the transforming entity's id, the ORDER BY was
the temporal index of the entity the *feature* came from. For a native feature
those are one entity. For a feature brought across a direct transfer — as-of
or plain — they are not, and the ORDER BY named a column the transforming
entity's synth CTE does not have::

    -- patients_transform (from patients_synth _ego)
    avg("ROLLING_MEAN_7(care_plans.cost)")
      over (partition by patient_id order by plan_date ...)    -- no plan_date here
    UndefinedColumn: column "plan_date" does not exist

The window now orders by the transforming entity's own temporal index. That is
also the semantics a transferred value calls for: after an as-of transfer the
value is already "the parent's state as of this row's date", so a rolling mean
over it is a rolling mean along the *target's* rows — not along the parent's
history, which the transfer has deliberately collapsed to one row.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from featurizer import Featurizer

CONFIG = """target: patients
max_depth: 2
intervals: []
aggregations: []
transformations: [identity, {tf}]
entities:
  - alias: patients
    id: patient_id
    table: patients
    temporal_ix: admission_date
    variables:
      age: {{type: numeric}}
  - alias: care_plans
    id: plan_id
    table: care_plans
    temporal_ix: plan_date
    variables:
      cost: {{type: numeric}}
relationships:
  - parent: {{entity: care_plans, key: patient_id}}
    child: {{entity: patients, key: patient_id}}
{temporal}"""

AS_OF = "    temporal: {mode: as_of, grace: P7D, child_timestamp: plan_date}\n"
WINDOWS = ["lag_1", "rolling_mean_7", "rolling_median_7", "cum_sum"]


def _write(tmp_path: Path, tf: str, temporal: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG.format(tf=tf, temporal=temporal))
    return str(path)


@pytest.mark.parametrize("tf", WINDOWS)
@pytest.mark.parametrize("temporal", [AS_OF, ""], ids=["as_of", "plain"])
def test_target_transform_orders_by_the_target_timeline(tmp_path, tf, temporal):
    query = Featurizer(_write(tmp_path, tf, temporal)).query
    target = query[query.index("patients_transform as (") :]
    assert "plan_date" not in target, (
        "the target's transform CTE must not order by the source entity's "
        "temporal index — patients_synth does not project it"
    )


@pytest.mark.parametrize("tf", WINDOWS)
def test_source_transform_still_orders_by_its_own_timeline(tmp_path, tf):
    """Native features are untouched: care_plans' own windows walk plan_date."""
    query = Featurizer(_write(tmp_path, tf, AS_OF)).query
    source = query[
        query.index("care_plans_transform as (") : query.index("patients_synth as (")
    ]
    assert "plan_date" in source


@pytest.mark.integration
@pytest.mark.parametrize("tf", WINDOWS)
@pytest.mark.parametrize("temporal", [AS_OF, ""], ids=["as_of", "plain"])
def test_window_over_a_transferred_feature_executes(tmp_path, tf, temporal):
    """Against PostgreSQL: before the fix every one of these raised UndefinedColumn."""
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    query = Featurizer(_write(tmp_path, tf, temporal)).query
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            "create temp table patients "
            "(patient_id int, admission_date date, age float8)"
        )
        cur.execute(
            "insert into patients values (1, date '2024-03-01', 40), "
            "(1, date '2024-04-01', 41), (2, date '2024-03-05', 55)"
        )
        # One plan per patient: a plain forward transfer is a many-to-one
        # lookup, and a non-unique parent key would fan the target out.
        cur.execute(
            "create temp table care_plans "
            "(plan_id int, patient_id int, plan_date date, cost float8)"
        )
        cur.execute(
            "insert into care_plans values (10, 1, date '2024-02-20', 250), "
            "(12, 2, date '2024-02-01', 175)"
        )
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute("insert into as_of_dates values (date '2024-06-01')")
        cur.execute(query)
        rows = cur.fetchall()
        conn.rollback()
    assert len(rows) == 3


AS_OF_NO_GRACE = "    temporal: {mode: as_of, child_timestamp: plan_date}\n"


@pytest.mark.integration
def test_window_over_a_transferred_value_walks_the_target_rows(tmp_path):
    """The values, not only the SQL: two visits, one care plan in force for both.

    ``CUM_SUM(care_plans.cost)`` is computed on the care-plan side, along the
    care-plan timeline, and transferred: one plan, so 250 on both visits.

    ``CUM_SUM(care_plans.CUM_SUM(care_plans.cost))`` is a window over that
    transferred value in the target's transform, so it walks the *visits*:
    250 on the first, 500 on the second. Along the care-plan timeline it could
    only ever see one row — the transfer collapses the plan's history to one
    row per visit — so 500 is the value that proves which timeline is walked.

    No ``grace`` here on purpose: ``grace: P7D`` renders a 7-day *lookback*
    cap (``plan_date >= admission_date - interval 'P7D'``), which would leave a
    plan dated before that window unmatched and every value NULL.
    """
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    query = Featurizer(_write(tmp_path, "cum_sum", AS_OF_NO_GRACE)).query
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            "create temp table patients "
            "(patient_id int, admission_date date, age float8)"
        )
        cur.execute(
            "insert into patients values (1, date '2024-03-01', 40), "
            "(1, date '2024-04-01', 41)"
        )
        cur.execute(
            "create temp table care_plans "
            "(plan_id int, patient_id int, plan_date date, cost float8)"
        )
        cur.execute("insert into care_plans values (10, 1, date '2024-02-20', 250)")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute("insert into as_of_dates values (date '2024-06-01')")
        cur.execute(query)
        columns = [d.name for d in cur.description]
        rows = sorted(cur.fetchall(), key=lambda r: r[columns.index("admission_date")])
        conn.rollback()

    def column(name: str) -> list:
        return [r[columns.index(name)] for r in rows]

    # Transferred as-is: the plan's own running total, one plan -> 250, 250.
    assert column("CUM_SUM(care_plans.cost)") == [250.0, 250.0]
    # Windowed in the target: walks the visits -> 250, 500.
    assert column("CUM_SUM(care_plans.CUM_SUM(care_plans.cost))") == [250.0, 500.0]
    # A native target feature, for contrast: already walked the visits.
    assert column("CUM_SUM(patients.age)") == [40.0, 81.0]
