"""Realistic value-assertions: DonorsChoose (DSSG Triage sample).

Complements the food inspections tier with what that dataset cannot cover:
a **timestamp** temporal_ix (``donation_timestamp`` — the column that exposed
bug #7, ``daterange @> timestamp``), a static child without temporal_ix
(``resources`` — must get all-time aggregations only), depth-3 nesting
(schools -> projects -> donations), and co-donation graph edges.

The project cohort is activity-weighted (most-donated first, ties by id) but
fully deterministic — the first-N-by-id cohort is nearly donation-free.
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

AS_OF_DATES = ["2012-01-01", "2012-07-01", "2013-01-01", "2013-07-01"]
MID = "2013-01-01"
COHORT_N = 100
SCHEMA = "donorschoose"

# Deterministic, activity-weighted cohort: most-donated projects first.
PROJECT_ORDER = (
    "(select count(*) from donorschoose.donations d "
    "where d.projectid = projects.projectid) desc, projectid"
)


def _run_projects(conn, n: int = COHORT_N) -> tuple[list[dict], str]:
    config = load_config("donorschoose_projects")
    cohort = make_cohort(
        conn,
        source_table=f"{SCHEMA}.projects",
        order_by=PROJECT_ORDER,
        n=n,
    )
    retarget(config, "projects", cohort)
    for child in ("donations", "resources"):
        subset = make_child_subset(
            conn,
            source_table=f"{SCHEMA}.{child}",
            key_col="projectid",
            cohort=cohort,
            cohort_key="projectid",
        )
        retarget(config, child, subset)
    make_as_of_dates(conn, AS_OF_DATES)
    return run_featurizer(conn, config), cohort


def _value(rows, as_of, projectid, col_substr):
    return feature(
        rows,
        as_of=as_of,
        id_col="projectid",
        entity_id=projectid,
        col_substr=col_substr,
    )


def test_grid_shape_and_no_dead_families(donorschoose_db):
    rows, _ = _run_projects(donorschoose_db)
    assert len(rows) == len(AS_OF_DATES) * COHORT_N

    families = [
        "COUNT(",
        "SUM(donations.donation_to_project)",
        "MEAN(",
        "MIN(",
        "MAX(",
        "STDDEV(",
        "P90(",
        "NUNIQUE(donations.donor_acctid)",
        "GAP_MEAN(",
        "EVENT_RATE(",
        "IN_DEGREE(",
        "OUT_DEGREE(",
        "DEGREE(",
        "K_HOP_2_COUNT(",
        "CLUSTERING_COEFF(",
    ]
    for family in families:
        columns = feature_columns(rows, family)
        assert columns, f"no output column for family {family!r}"
        alive = any(row[col] is not None for row in rows for col in columns)
        assert alive, f"family {family!r} is all-NULL on real data"


def test_numeric_aggregations_match_sql(donorschoose_db):
    """SUM / MEAN / MAX / STDDEV / P90 of donation amounts equal independent
    recomputations, causally cut at the as-of date."""
    rows, cohort = _run_projects(donorschoose_db)
    with donorschoose_db.cursor() as cur:
        cur.execute(
            f"""
            select d.projectid from {SCHEMA}.donations d
            join {cohort} c using (projectid)
            where d.donation_timestamp <= %s
            group by d.projectid having count(*) >= 4
            order by count(*) desc, d.projectid limit 5
            """,
            (MID,),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled

    recompute = f"""
        select {{expr}} from {SCHEMA}.donations
        where projectid = %s and donation_timestamp <= %s
    """
    checks = [
        ("SUM(donations.donation_to_project)", "sum(donation_to_project)", 1e-9),
        ("MEAN(donations.donation_to_project)", "avg(donation_to_project)", 1e-9),
        ("MAX(donations.donation_to_project)", "max(donation_to_project)", 1e-9),
        ("STDDEV(donations.donation_to_project)", "stddev(donation_to_project)", 1e-9),
        (
            "P90(donations.donation_to_project)",
            "percentile_cont(0.9) within group (order by donation_to_project)",
            1e-6,
        ),
    ]
    for projectid in sampled:
        for col, expr, tol in checks:
            got = _value(rows, MID, projectid, col)
            want = expect_sql(
                donorschoose_db, recompute.format(expr=expr), (projectid, MID)
            )
            assert got is not None and want is not None
            assert math.isclose(
                float(got), float(want), rel_tol=tol
            ), f"{col} for project {projectid}: featurizer={got} sql={want}"


def test_nunique_donors_is_causal(donorschoose_db):
    """NUNIQUE over donor ids at as-of T counts only donors who had donated by
    T — donations after T must be excluded."""
    rows, cohort = _run_projects(donorschoose_db)
    with donorschoose_db.cursor() as cur:
        cur.execute(
            f"""
            select c.projectid from {cohort} c
            where exists (select 1 from {SCHEMA}.donations x
                          where x.projectid = c.projectid
                            and x.donation_timestamp <= %s)
              and exists (select 1 from {SCHEMA}.donations y
                          where y.projectid = c.projectid
                            and y.donation_timestamp > %s)
            order by c.projectid limit 5
            """,
            (MID, MID),
        )
        straddlers = [r[0] for r in cur.fetchall()]
    assert straddlers, "no cohort project has donations on both sides of MID"

    for projectid in straddlers:
        got = _value(rows, MID, projectid, "NUNIQUE(donations.donor_acctid)")
        want_causal = expect_sql(
            donorschoose_db,
            f"select count(distinct donor_acctid) from {SCHEMA}.donations "
            "where projectid = %s and donation_timestamp <= %s",
            (projectid, MID),
        )
        want_uncut = expect_sql(
            donorschoose_db,
            f"select count(distinct donor_acctid) from {SCHEMA}.donations "
            "where projectid = %s",
            (projectid,),
        )
        assert int(got) == want_causal
        # The whole point: the un-cut count would be different (or at least
        # the donation rows after MID exist), so equality with the causal
        # count is meaningful.
        assert want_causal <= want_uncut


def test_static_resources_get_all_time_aggregations_only(donorschoose_db):
    """A child without temporal_ix is aggregated all-time; the planner must
    not emit interval-windowed columns for it."""
    rows, cohort = _run_projects(donorschoose_db)

    interval_cols = [
        col
        for col in rows[0]
        if col.startswith("SUM(resources.") and "|interval=" in col
    ]
    assert not interval_cols, f"static child got interval windows: {interval_cols!r}"

    with donorschoose_db.cursor() as cur:
        cur.execute(f"""
            select r.projectid from {SCHEMA}.resources r
            join {cohort} c using (projectid)
            group by r.projectid having count(*) >= 2
            order by r.projectid limit 3
            """)
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled

    for projectid in sampled:
        got = _value(rows, MID, projectid, "SUM(resources.item_quantity)")
        want = expect_sql(
            donorschoose_db,
            f"select sum(item_quantity) from {SCHEMA}.resources where projectid = %s",
            (projectid,),
        )
        assert got is not None
        assert math.isclose(float(got), float(want), rel_tol=1e-9)


def test_depth3_schools_nesting_matches_two_level_sql(donorschoose_db):
    """schools -> projects -> donations at max_depth=3: the stacked feature
    SUM(projects.SUM(donations.donation_to_project)) equals a hand-written
    two-level aggregation with both causal cuts applied."""
    config = load_config("donorschoose_schools_depth3")
    cohort = make_cohort(
        donorschoose_db,
        source_table=f"{SCHEMA}.schools",
        order_by="n_projects desc, schoolid",
        n=10,
    )
    retarget(config, "schools", cohort)
    projects = make_child_subset(
        donorschoose_db,
        source_table=f"{SCHEMA}.projects",
        key_col="schoolid",
        cohort=cohort,
        cohort_key="schoolid",
    )
    retarget(config, "projects", projects)
    donations = make_child_subset(
        donorschoose_db,
        source_table=f"{SCHEMA}.donations",
        key_col="projectid",
        cohort=projects,
        cohort_key="projectid",
    )
    retarget(config, "donations", donations)
    as_ofs = ["2012-07-01", "2013-07-01"]
    make_as_of_dates(donorschoose_db, as_ofs)
    rows = run_featurizer(donorschoose_db, config)
    assert len(rows) == len(as_ofs) * 10

    nested_col = "SUM(projects.SUM(donations.donation_to_project))"
    assert feature_columns(rows, nested_col) == [nested_col]

    with donorschoose_db.cursor() as cur:
        cur.execute(
            f"""
            select p.schoolid from {SCHEMA}.donations d
            join {SCHEMA}.projects p using (projectid)
            join {cohort} c on c.schoolid = p.schoolid
            where d.donation_timestamp <= %s and p.date_posted <= %s
            group by p.schoolid order by count(*) desc, p.schoolid limit 3
            """,
            (as_ofs[1], as_ofs[1]),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled

    for schoolid in sampled:
        got = feature(
            rows,
            as_of=as_ofs[1],
            id_col="schoolid",
            entity_id=schoolid,
            col_substr=nested_col,
        )
        want = expect_sql(
            donorschoose_db,
            f"""
            select sum(d.donation_to_project)
            from {SCHEMA}.donations d
            join {SCHEMA}.projects p using (projectid)
            where p.schoolid = %s
              and p.date_posted <= %s
              and d.donation_timestamp <= %s
            """,
            (schoolid, as_ofs[1], as_ofs[1]),
        )
        assert got is not None and want is not None
        assert math.isclose(
            float(got), float(want), rel_tol=1e-9
        ), f"school {schoolid}: featurizer={got} sql={want}"


def test_k_hop_2_matches_sql_recomputation(donorschoose_db):
    """K_HOP_2_COUNT over co-donation edges equals an independent two-hop SQL
    recomputation (distinct nodes at exactly distance 2, causally bounded)."""
    rows, cohort = _run_projects(donorschoose_db)
    with donorschoose_db.cursor() as cur:
        cur.execute(
            f"""
            select distinct c.projectid from {cohort} c
            join {SCHEMA}.project_edges e
              on c.projectid in (e.source_project, e.target_project)
             and e.knowable_at <= %s
            order by c.projectid limit 5
            """,
            (MID,),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled

    k2_sql = f"""
        with und as (
            select source_project as a, target_project as b
            from {SCHEMA}.project_edges where knowable_at <= %(as_of)s
            union
            select target_project, source_project
            from {SCHEMA}.project_edges where knowable_at <= %(as_of)s
        )
        select count(distinct two.b) from und one
        join und two on two.a = one.b
        where one.a = %(ego)s
          and two.b <> %(ego)s
          and two.b not in (select b from und where a = %(ego)s)
    """
    for projectid in sampled:
        got = _value(rows, MID, projectid, "K_HOP_2_COUNT(projects.project_edges)")
        with donorschoose_db.cursor() as cur:
            cur.execute(k2_sql, {"as_of": MID, "ego": projectid})
            want = cur.fetchone()[0]
        if want == 0:
            assert got is None or int(got) == 0
        else:
            assert (
                got is not None and int(got) == want
            ), f"project {projectid}: featurizer={got} sql={want}"


def test_graph_co_donation_degree_is_causal(donorschoose_db):
    rows, cohort = _run_projects(donorschoose_db)
    with donorschoose_db.cursor() as cur:
        cur.execute(f"""
            select distinct c.projectid from {cohort} c
            join {SCHEMA}.project_edges e
              on c.projectid in (e.source_project, e.target_project)
            order by c.projectid limit 5
            """)
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled

    early, late = AS_OF_DATES[0], AS_OF_DATES[-1]
    degree_sql = (
        f"select count(*) from {SCHEMA}.project_edges "
        "where (source_project = %s or target_project = %s) "
        "and knowable_at <= %s"
    )
    for projectid in sampled:
        for as_of in (early, late):
            got = _value(rows, as_of, projectid, "DEGREE(projects.project_edges)")
            want = expect_sql(
                donorschoose_db, degree_sql, (projectid, projectid, as_of)
            )
            if want == 0:
                assert got is None or int(got) == 0
            else:
                assert got is not None and int(got) == want
