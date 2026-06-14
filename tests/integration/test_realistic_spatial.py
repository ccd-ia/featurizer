"""Value-assertions for the M1d spatial second-table pass.

Runs over the Chicago Food Inspections cohort with a lat/lon ``spatial_ix``
injected on ``facilities`` and a self spatial relationship (facilities near
facilities). ``COLOCATION_COUNT`` and ``DISTANCE_TO_NEAREST`` are checked
against an independent haversine recomputation over the same cohort, with the
``first_seen <= as_of`` membership bound asserted directly.
"""

from __future__ import annotations

import math

import pytest

from ._harness import run_featurizer
from ._realistic import (
    expect_sql,
    feature,
    feature_columns,
    load_config,
    make_as_of_dates,
    make_child_subset,
    make_cohort,
    retarget,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SCHEMA = "food_inspections"
AS_OF_DATES = ["2015-01-01", "2015-07-01", "2016-07-01"]
MID = "2015-07-01"
COHORT_N = 60
WITHIN_M = 50000  # ~Chicago-wide, so the cohort has in-radius neighbours


def _hav(ego: str, other: str) -> str:
    """Great-circle distance (m) between two aliases' latitude/longitude — the
    exact formula featurizer emits (ego = lat1, other = lat2)."""
    return (
        f"2 * 6371000 * asin(sqrt("
        f"power(sin(radians({other}.latitude - {ego}.latitude) / 2), 2) "
        f"+ cos(radians({ego}.latitude)) * cos(radians({other}.latitude)) "
        f"* power(sin(radians({other}.longitude - {ego}.longitude) / 2), 2)))"
    )


def _run_spatial_cohort(conn, n: int = COHORT_N):
    config = load_config("food_inspections", aggregations=["count"], intervals=[])
    for entity in config["entities"]:
        if entity["alias"] == "facilities":
            entity["spatial_ix"] = {"lat": "latitude", "lon": "longitude"}
    config["spatial_relationships"] = [
        {
            "name": "nearby",
            "left": "facilities",
            "right": "facilities",
            "within_m": WITHIN_M,
            "bandwidth_m": 10000,
        }
    ]
    cohort = make_cohort(
        conn, source_table=f"{SCHEMA}.facilities", order_by="license_no", n=n
    )
    retarget(config, "facilities", cohort)
    for child in ("inspections", "licenses"):
        subset = make_child_subset(
            conn,
            source_table=f"{SCHEMA}.{child}",
            key_col="license_no",
            cohort=cohort,
            cohort_key="license_no",
        )
        retarget(config, child, subset)
    make_as_of_dates(conn, AS_OF_DATES)
    return run_featurizer(conn, config), cohort


def _value(rows, license_no, col_substr, as_of=MID):
    return feature(
        rows,
        as_of=as_of,
        id_col="license_no",
        entity_id=license_no,
        col_substr=col_substr,
    )


def test_spatial_families_alive(food_db):
    """All three spatial families produce a non-NULL value somewhere."""
    rows, _ = _run_spatial_cohort(food_db)
    assert len(rows) == len(AS_OF_DATES) * COHORT_N
    for family in (
        "COLOCATION_COUNT(nearby)",
        "DISTANCE_TO_NEAREST(nearby)",
        "KDE_INTENSITY(nearby)",
    ):
        columns = feature_columns(rows, family)
        assert columns, f"no output column for {family!r}"
        assert any(
            row[col] is not None for row in rows for col in columns
        ), f"spatial family {family!r} is all-NULL on real data"


def test_colocation_count_matches_haversine_recompute(food_db):
    """COLOCATION_COUNT equals an independent in-radius haversine count over the
    cohort, bounded by first_seen <= as_of and excluding the ego."""
    rows, cohort = _run_spatial_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(f"""
            select license_no from {cohort}
            where latitude is not null and longitude is not null
            order by license_no limit 8
            """)
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "cohort has no facilities with coordinates"

    saw_positive = False
    for license_no in sampled:
        got = _value(rows, license_no, "COLOCATION_COUNT(nearby)")
        want = expect_sql(
            food_db,
            f"""
            select count(*)
            from {cohort} ego, {cohort} r
            where ego.license_no = %s
              and r.license_no <> ego.license_no
              and r.first_seen <= %s
              and {_hav("ego", "r")} <= %s
            """,
            (license_no, MID, WITHIN_M),
        )
        assert int(got or 0) == int(want)
        saw_positive = saw_positive or want > 0
    assert saw_positive, "no sampled facility had an in-radius neighbour"


def test_distance_to_nearest_matches_min_haversine(food_db):
    """DISTANCE_TO_NEAREST equals the minimum in-radius haversine to another
    cohort facility knowable as-of the cutoff."""
    rows, cohort = _run_spatial_cohort(food_db)
    with food_db.cursor() as cur:
        # Facilities that actually have an in-radius, as-of-visible neighbour.
        cur.execute(
            f"""
            select ego.license_no
            from {cohort} ego
            where ego.latitude is not null and ego.longitude is not null
              and exists (
                select 1 from {cohort} r
                where r.license_no <> ego.license_no and r.first_seen <= %s
                  and {_hav("ego", "r")} <= %s
              )
            order by ego.license_no limit 5
            """,
            (MID, WITHIN_M),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "no cohort facility has an in-radius visible neighbour"

    for license_no in sampled:
        got = _value(rows, license_no, "DISTANCE_TO_NEAREST(nearby)")
        want = expect_sql(
            food_db,
            f"""
            select min({_hav("ego", "r")})
            from {cohort} ego, {cohort} r
            where ego.license_no = %s
              and r.license_no <> ego.license_no
              and r.first_seen <= %s
              and {_hav("ego", "r")} <= %s
            """,
            (license_no, MID, WITHIN_M),
        )
        assert got is not None and want is not None
        assert math.isclose(float(got), float(want), rel_tol=1e-9)


def test_colocation_is_causal_over_time(food_db):
    """Co-location count is monotonic non-decreasing as the as-of date advances
    (more facilities become knowable), for a facility with neighbours."""
    rows, cohort = _run_spatial_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(
            f"""
            select ego.license_no
            from {cohort} ego
            where ego.latitude is not null and ego.longitude is not null
              and exists (
                select 1 from {cohort} r
                where r.license_no <> ego.license_no
                  and r.first_seen <= %s
                  and {_hav("ego", "r")} <= %s
              )
            order by ego.license_no limit 3
            """,
            (AS_OF_DATES[0], WITHIN_M),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "no facility has an early-visible neighbour"

    for license_no in sampled:
        early = _value(
            rows, license_no, "COLOCATION_COUNT(nearby)", as_of=AS_OF_DATES[0]
        )
        late = _value(
            rows, license_no, "COLOCATION_COUNT(nearby)", as_of=AS_OF_DATES[-1]
        )
        assert int(early or 0) <= int(late or 0)
