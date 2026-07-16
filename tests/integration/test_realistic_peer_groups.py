"""Value-assertions for the M1d peer-group pass (Option A: by categorical column).

Two tiers run here:

- *Realistic* (``food_db``): peer features over the Chicago Food Inspections
  cohort grouped by ``facility_type``. ``facilities`` has no numeric attribute,
  so this tier asserts the cross-stream ``PEER_EVENT_RATE`` (mean per-peer
  inspection count) and ``PEER_GROUP_SIZE`` against independent recomputations,
  with the leave-one-out and ``<= as_of_date`` invariants checked directly.

- *Synthetic* (``pg_conn``): a tiny hand-computed fixture with a numeric
  measure and a child stream, so the measure statistics (mean / z-score /
  percentile / ego-minus-peer delta) are checked against exact constants.
"""

from __future__ import annotations

import math

import pytest

from ._harness import create_temp_table, run_featurizer
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

pytestmark = [pytest.mark.integration]

SCHEMA = "food_inspections"
AS_OF_DATES = ["2015-01-01", "2015-07-01", "2016-07-01"]
MID = "2015-07-01"
COHORT_N = 50


def _inject_peer_group(config: dict) -> None:
    for entity in config["entities"]:
        if entity["alias"] == "facilities":
            entity["peer_groups"] = [{"by": "facility_type"}]
            return
    raise AssertionError("facilities entity not found in config")


def _run_peer_cohort(conn, n: int = COHORT_N):
    config = load_config("food_inspections", aggregations=["count"], intervals=[])
    _inject_peer_group(config)
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


@pytest.mark.slow
def test_peer_families_alive_on_real_data(food_db):
    """Both peer families produce a non-NULL value somewhere on the cohort."""
    rows, _ = _run_peer_cohort(food_db)
    assert len(rows) == len(AS_OF_DATES) * COHORT_N
    for family in ("PEER_GROUP_SIZE(", "PEER_EVENT_RATE("):
        columns = feature_columns(rows, family)
        assert columns, f"no output column for peer family {family!r}"
        assert any(row[col] is not None for row in rows for col in columns), (
            f"peer family {family!r} is all-NULL on real data"
        )


@pytest.mark.slow
def test_peer_group_size_is_leave_one_out_and_causal(food_db):
    """PEER_GROUP_SIZE equals (#same-type facilities knowable as-of T) minus the
    ego itself when the ego is also knowable — recomputed over the cohort."""
    rows, cohort = _run_peer_cohort(food_db)
    with food_db.cursor() as cur:
        # Cohort facilities whose facility_type group (first_seen <= MID) has at
        # least two members — so the leave-one-out size is positive.
        cur.execute(
            f"""
            select f.license_no
            from {cohort} f
            join {cohort} g
              on g.facility_type = f.facility_type and g.first_seen <= %s
            where f.first_seen <= %s
            group by f.license_no
            having count(*) >= 2
            order by f.license_no limit 5
            """,
            (MID, MID),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "no cohort facility shares its type with another (visible) one"

    for license_no in sampled:
        got = _value(rows, license_no, "PEER_GROUP_SIZE(")
        want = expect_sql(
            food_db,
            f"""
            select (
                select count(*) from {cohort} p
                where p.facility_type = ego.facility_type and p.first_seen <= %s
            ) - (case when ego.first_seen <= %s then 1 else 0 end)
            from {cohort} ego where ego.license_no = %s
            """,
            (MID, MID, license_no),
        )
        assert int(got) == int(want)


@pytest.mark.slow
def test_peer_event_rate_matches_leave_one_out_mean(food_db):
    """PEER_EVENT_RATE equals the mean (over same-type peers, excluding the ego,
    knowable as-of T) of each peer's inspection count knowable as-of T."""
    rows, cohort = _run_peer_cohort(food_db)
    subset = "subset_inspections"  # created by _run_peer_cohort via make_child_subset
    with food_db.cursor() as cur:
        cur.execute(
            f"""
            select f.license_no
            from {cohort} f
            join {cohort} g
              on g.facility_type = f.facility_type and g.first_seen <= %s
             and g.license_no <> f.license_no
            where f.first_seen <= %s
            group by f.license_no
            having count(*) >= 2
            order by f.license_no limit 5
            """,
            (MID, MID),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "need cohort facilities with >= 2 visible same-type peers"

    for license_no in sampled:
        got = _value(rows, license_no, "PEER_EVENT_RATE(")
        want = expect_sql(
            food_db,
            f"""
            select avg(cnt)::float from (
                select (
                    select count(*) from {subset} i
                    where i.license_no = p.license_no and i.inspection_date <= %s
                ) as cnt
                from {cohort} p
                where p.facility_type = (
                          select facility_type from {cohort} where license_no = %s
                      )
                  and p.first_seen <= %s
                  and p.license_no <> %s
            ) q
            """,
            (MID, license_no, MID, license_no),
        )
        assert got is not None and want is not None
        assert math.isclose(float(got), float(want), rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Synthetic fixture: exact constants for the measure statistics.
# --------------------------------------------------------------------------- #

PEER_CONFIG = {
    "target": "ent",
    "max_depth": 2,
    "intervals": [],
    "aggregations": ["count"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "ent",
            "table": "ent",
            "id": "ent_id",
            "temporal_ix": "born",
            "variables": {
                "grp": {"type": "categorical"},
                "m": {"type": "numeric"},
            },
            "peer_groups": [{"by": "grp", "measures": ["m"]}],
        },
        {
            "alias": "ev",
            "table": "ev",
            "id": "ev_id",
            "temporal_ix": "ts",
            "variables": {"ent_id": {"type": "index"}},
        },
    ],
    # Parent and child share the join-key *name* (ent_id) — the aggregation CTE
    # assumes that, as the other realistic configs do.
    "relationships": [
        {
            "parent": {"entity": "ent", "key": "ent_id"},
            "child": {"entity": "ev", "key": "ent_id"},
        }
    ],
}

AS_OF = "2020-06-01"


def _seed_synthetic(conn) -> None:
    create_temp_table(
        conn,
        "ent",
        [
            ("ent_id", "int"),
            ("grp", "text"),
            ("m", "double precision"),
            ("born", "date"),
        ],
        [
            (1, "A", 10.0, "2019-01-01"),
            (2, "A", 20.0, "2019-01-01"),
            (3, "A", 30.0, "2019-01-01"),
            (4, "A", 40.0, "2021-01-01"),  # born after as-of -> not a peer
            (5, "B", 100.0, "2019-01-01"),  # singleton group
        ],
    )
    create_temp_table(
        conn,
        "ev",
        [("ev_id", "int"), ("ent_id", "int"), ("ts", "date")],
        [
            (1, 1, "2020-01-01"),
            (2, 1, "2020-02-01"),  # ent 1 -> 2 events <= as-of
            (3, 2, "2019-01-01"),
            (4, 2, "2019-02-01"),
            (5, 2, "2019-03-01"),
            (6, 2, "2019-04-01"),  # ent 2 -> 4 events <= as-of
            (7, 2, "2021-01-01"),  # future -> excluded by the causal bound
            # ent 3 -> 0 events
        ],
    )
    make_as_of_dates(conn, [AS_OF])


def _syn_value(rows, ent_id, col_substr):
    return feature(
        rows, as_of=AS_OF, id_col="ent_id", entity_id=ent_id, col_substr=col_substr
    )


def test_peer_measure_statistics_exact(pg_conn):
    """Mean / delta / z-score / percentile / size / event-rate against constants.

    Group A members knowable as-of 2020-06-01 are {1: m=10, 2: m=20, 3: m=30}
    (id 4 is born later). For ego id=1, peers = {2, 3}:
      mean=25, delta=10-25=-15, std(sample)=sqrt(50), z=-15/sqrt(50),
      pctile = |peers with m<10| / 2 = 0, size = 2,
      event_rate = mean(events of {2,3}) = (4 + 0)/2 = 2.
    """
    _seed_synthetic(pg_conn)
    rows = run_featurizer(pg_conn, PEER_CONFIG)
    assert len(rows) == 5  # one as-of x five entities

    # Exact column names (PEER_MEAN( is also a substring of EGO_MINUS_PEER_MEAN().
    mean = "PEER_MEAN(ent.m by grp)"
    delta = "EGO_MINUS_PEER_MEAN(ent.m by grp)"
    zscore = "PEER_ZSCORE(ent.m by grp)"
    pctile = "PEER_PCTILE(ent.m by grp)"
    size = "PEER_GROUP_SIZE(ent by grp)"
    rate = "PEER_EVENT_RATE(ent.ev by grp)"

    # Ego id=1 (m=10), peers {2 (m=20), 3 (m=30)}.
    assert math.isclose(float(_syn_value(rows, 1, mean)), 25.0, rel_tol=1e-9)
    assert math.isclose(float(_syn_value(rows, 1, delta)), -15.0, rel_tol=1e-9)
    assert math.isclose(
        float(_syn_value(rows, 1, zscore)), -15.0 / math.sqrt(50.0), rel_tol=1e-9
    )
    assert math.isclose(float(_syn_value(rows, 1, pctile)), 0.0, abs_tol=1e-12)
    assert int(_syn_value(rows, 1, size)) == 2
    assert math.isclose(float(_syn_value(rows, 1, rate)), 2.0, rel_tol=1e-9)

    # Ego id=3 (m=30): peers {1 (m=10), 2 (m=20)} -> mean 15, both below -> pctile 1.
    assert math.isclose(float(_syn_value(rows, 3, mean)), 15.0, rel_tol=1e-9)
    assert math.isclose(float(_syn_value(rows, 3, pctile)), 1.0, rel_tol=1e-9)

    # Singleton group B (id=5): leave-one-out set is empty -> NULL, never a crash.
    assert _syn_value(rows, 5, size) == 0
    assert _syn_value(rows, 5, mean) is None


def test_peer_membership_excludes_future_born(pg_conn):
    """Id 4 (born 2021) is not a peer at the 2020 as-of, so group A size is 2
    for its members and id 4's own peer set is {1,2,3}."""
    _seed_synthetic(pg_conn)
    rows = run_featurizer(pg_conn, PEER_CONFIG)
    # Members 1..3 see each other (size 2 each); id 4 sees all three (size 3,
    # since it is itself not counted as knowable).
    assert int(_syn_value(rows, 2, "PEER_GROUP_SIZE(")) == 2
    assert int(_syn_value(rows, 4, "PEER_GROUP_SIZE(")) == 3
