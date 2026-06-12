"""Realistic value-assertions: Chicago Food Inspections.

Every test runs the same config (``datasets/configs/food_inspections.yaml``)
over a deterministic 50-facility cohort and compares featurizer's output
against *independent* SQL recomputations on the seeded schema. The causal
invariant — features at as-of T are computed only from rows with
``timestamp <= T`` — is asserted directly, not assumed.

Children are restricted to the cohort via TEMP subsets (see
``_realistic.make_child_subset``): correlated SubqueryAggregator features
rescan the un-indexed child CTE per group, which is unusably slow on the full
65k-row table. Values are unchanged because the subset contains every row of
each cohort member. ``test_full_table_cheap_aggregations_smoke`` covers the
full-scale path with cheap aggregators only.
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

AS_OF_DATES = ["2015-01-01", "2015-07-01", "2016-07-01"]
MID = "2015-07-01"
COHORT_N = 50
SCHEMA = "food_inspections"


def _run_cohort(conn, n: int = COHORT_N) -> tuple[list[dict], str]:
    """Run the food config over a deterministic cohort; return (rows, cohort)."""
    config = load_config("food_inspections")
    cohort = make_cohort(
        conn,
        source_table=f"{SCHEMA}.facilities",
        order_by="license_no",
        n=n,
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


def _value(rows, as_of, license_no, col_substr):
    return feature(
        rows,
        as_of=as_of,
        id_col="license_no",
        entity_id=license_no,
        col_substr=col_substr,
    )


def test_grid_shape_and_no_dead_families(food_db):
    """Output is the full |as_of| x |cohort| grid and every requested feature
    family produces at least one non-NULL value somewhere on real data."""
    rows, _ = _run_cohort(food_db)
    assert len(rows) == len(AS_OF_DATES) * COHORT_N

    families = [
        "COUNT(",
        "SUM(",
        "MEAN(",
        "MAX(",
        "NUNIQUE(",
        "MODE(",
        "ENTROPY(",  # also matches SEQUENCE_ENTROPY — fine for liveness
        "HHI(",
        "GAP_MEAN(",
        "GAP_STDDEV(",
        "GAP_CV(",
        "SEQUENCE_ENTROPY(",
        "LONGEST_STREAK(",
        "RECENCY(",
        "IN_DEGREE(",
        "OUT_DEGREE(",
        "DEGREE(",
        "CLUSTERING_COEFF(",
        "COMMON_NEIGHBOURS_MEAN(",
        "JACCARD_MEAN(",
        "ADAMIC_ADAR_MEAN(",
        "application_type",  # the as-of pull from licenses
    ]
    for family in families:
        columns = feature_columns(rows, family)
        assert columns, f"no output column for family {family!r}"
        alive = any(row[col] is not None for row in rows for col in columns)
        assert alive, f"family {family!r} is all-NULL on real data"


def test_count_and_interval_causality(food_db):
    """COUNT at as-of T equals a manual recount of rows with date <= T, and
    the P1Y interval equals the manual windowed recount."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(
            f"""
            select i.license_no from {SCHEMA}.inspections i
            join {cohort} c using (license_no)
            where i.inspection_date <= %s
            group by i.license_no order by count(*) desc, i.license_no limit 5
            """,
            (MID,),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "no cohort facility has inspections before the mid as-of"

    for license_no in sampled:
        got_all = _value(rows, MID, license_no, "COUNT(inspections.inspection_id)")
        want_all = expect_sql(
            food_db,
            f"select count(*) from {SCHEMA}.inspections "
            "where license_no = %s and inspection_date <= %s",
            (license_no, MID),
        )
        assert int(got_all) == want_all

        got_p1y = _value(
            rows, MID, license_no, "COUNT(inspections.inspection_id|interval=P1Y)"
        )
        # The generated window is daterange((as_of - P1Y)::date, as_of, '[]')
        # — both endpoints inclusive.
        want_p1y = expect_sql(
            food_db,
            f"select count(*) from {SCHEMA}.inspections "
            "where license_no = %s "
            "and inspection_date >= (%s::date - interval 'P1Y')::date "
            "and inspection_date <= %s",
            (license_no, MID, MID),
        )
        assert int(got_p1y) == want_p1y
        assert int(got_p1y) <= int(got_all)


def test_categorical_entropy_hhi_mode_match_sql(food_db):
    """ENTROPY / HHI / MODE over inspection results equal independent
    recomputations on the seeded schema, causally cut at the as-of date."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(
            f"""
            select i.license_no from {SCHEMA}.inspections i
            join {cohort} c using (license_no)
            where i.inspection_date <= %s
            group by i.license_no having count(*) >= 3
            order by i.license_no limit 3
            """,
            (MID,),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled

    for license_no in sampled:
        got_entropy = _value(rows, MID, license_no, "ENTROPY(inspections.results)")
        want_entropy = expect_sql(
            food_db,
            f"""
            select -sum((freq::float / total) * ln(freq::float / total))
            from (
                select count(*) as freq, sum(count(*)) over () as total
                from {SCHEMA}.inspections
                where license_no = %s and inspection_date <= %s
                group by results
            ) d
            """,
            (license_no, MID),
        )
        assert math.isclose(float(got_entropy), float(want_entropy), rel_tol=1e-9)

        got_hhi = _value(rows, MID, license_no, "HHI(inspections.results)")
        want_hhi = expect_sql(
            food_db,
            f"""
            select sum(power(freq::float / total, 2))
            from (
                select count(*) as freq, sum(count(*)) over () as total
                from {SCHEMA}.inspections
                where license_no = %s and inspection_date <= %s
                group by results
            ) d
            """,
            (license_no, MID),
        )
        assert math.isclose(float(got_hhi), float(want_hhi), rel_tol=1e-9)

        got_mode = _value(rows, MID, license_no, "MODE(inspections.results)")
        # Recompute the set of maximal-frequency results; the generated
        # mode() may resolve ties either way.
        with food_db.cursor() as cur:
            cur.execute(
                f"""
                select results from {SCHEMA}.inspections
                where license_no = %s and inspection_date <= %s
                group by results
                having count(*) = (
                    select max(c) from (
                        select count(*) as c from {SCHEMA}.inspections
                        where license_no = %s and inspection_date <= %s
                        group by results
                    ) m
                )
                """,
                (license_no, MID, license_no, MID),
            )
            modal_set = {r[0] for r in cur.fetchall()}
        assert got_mode in modal_set


def test_gap_statistics_match_lag_sql(food_db):
    """gap_mean / gap_stddev equal a manual LAG() recomputation over the
    facility's inspection dates at or before the as-of date."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(
            f"""
            select i.license_no from {SCHEMA}.inspections i
            join {cohort} c using (license_no)
            where i.inspection_date <= %s
            group by i.license_no
            having count(*) >= 3 and count(*) = count(distinct i.inspection_date)
            order by i.license_no limit 3
            """,
            (MID,),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "need cohort facilities with >=3 distinct-date inspections"

    gap_sql = f"""
        select {{agg}}(gap) from (
            select inspection_date
                   - lag(inspection_date) over (order by inspection_date) as gap
            from {SCHEMA}.inspections
            where license_no = %s and inspection_date <= %s
        ) g where gap is not null
    """
    for license_no in sampled:
        got_mean = _value(
            rows, MID, license_no, "GAP_MEAN(inspections.inspection_date)"
        )
        want_mean = expect_sql(food_db, gap_sql.format(agg="avg"), (license_no, MID))
        assert math.isclose(float(got_mean), float(want_mean), rel_tol=1e-9)

        got_sd = _value(
            rows, MID, license_no, "GAP_STDDEV(inspections.inspection_date)"
        )
        want_sd = expect_sql(food_db, gap_sql.format(agg="stddev"), (license_no, MID))
        if want_sd is None:
            assert got_sd is None
        else:
            assert math.isclose(float(got_sd), float(want_sd), rel_tol=1e-9)


def test_m1c_markov_features_match_python_recomputation(food_db):
    """The M1c sequence features over inspection results equal an independent
    *Python* recomputation of the transition matrix — a genuinely different
    implementation than the generated SQL."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(
            f"""
            select i.license_no from {SCHEMA}.inspections i
            join {cohort} c using (license_no)
            where i.inspection_date <= %s
            group by i.license_no
            having count(*) >= 4 and count(*) = count(distinct i.inspection_date)
            order by count(*) desc, i.license_no limit 3
            """,
            (MID,),
        )
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "need cohort facilities with >=4 distinct-date inspections"

    for license_no in sampled:
        with food_db.cursor() as cur:
            cur.execute(
                f"select results, inspection_date from {SCHEMA}.inspections "
                "where license_no = %s and inspection_date <= %s "
                "order by inspection_date",
                (license_no, MID),
            )
            sequence = cur.fetchall()
        states = [r[0] for r in sequence]
        dates = [r[1] for r in sequence]

        # Transition matrix in Python.
        transitions = list(zip(states, states[1:]))
        assert transitions
        from collections import Counter

        joint = Counter(transitions)
        row_totals = Counter(prev for prev, _ in transitions)
        total = len(transitions)
        want_entropy = -sum(
            (freq / total) * math.log(freq / row_totals[prev])
            for (prev, _), freq in joint.items()
        )
        want_max_prob = max(
            freq / row_totals[prev] for (prev, _), freq in joint.items()
        )

        got_entropy = _value(
            rows, MID, license_no, "MARKOV_CONDITIONAL_ENTROPY(inspections.results)"
        )
        got_max_prob = _value(
            rows, MID, license_no, "MAX_TRANSITION_PROB(inspections.results)"
        )
        assert math.isclose(
            float(got_entropy), want_entropy, rel_tol=1e-9, abs_tol=1e-12
        )
        assert math.isclose(float(got_max_prob), want_max_prob, rel_tol=1e-9)

        # Recurrence interval: mean gap between same-state occurrences.
        same_state_gaps = []
        last_seen: dict = {}
        for state, when in zip(states, dates):
            if state in last_seen:
                same_state_gaps.append((when - last_seen[state]).days)
            last_seen[state] = when
        got_recurrence = _value(
            rows, MID, license_no, "RECURRENCE_INTERVAL(inspections.results)"
        )
        if same_state_gaps:
            want_recurrence = sum(same_state_gaps) / len(same_state_gaps)
            assert math.isclose(float(got_recurrence), want_recurrence, rel_tol=1e-9)
        else:
            assert got_recurrence is None

        # First passage to the configured target state ('Fail').
        got_fpt = _value(
            rows, MID, license_no, "FIRST_PASSAGE_TIME(inspections.results)"
        )
        fail_dates = [when for state, when in zip(states, dates) if state == "Fail"]
        if fail_dates:
            want_fpt = (min(fail_dates) - dates[0]).days
            assert int(got_fpt) == want_fpt
        else:
            assert got_fpt is None


def test_asof_license_join_pulls_latest_in_grace(food_db):
    """The as-of joined application_type is the latest license record with
    license_start_date <= first_seen within the P2Y grace window."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(f"""
            select c.license_no, c.first_seen from {cohort} c
            where exists (
                select 1 from {SCHEMA}.licenses l
                where l.license_no = c.license_no
                  and l.license_start_date <= c.first_seen
                  and l.license_start_date >= c.first_seen - interval 'P2Y'
            )
            order by c.license_no limit 5
            """)
        sampled = cur.fetchall()
    assert sampled, "no cohort facility has an in-grace license record"

    for license_no, first_seen in sampled:
        got = _value(rows, MID, license_no, "application_type")
        with food_db.cursor() as cur:
            cur.execute(
                f"""
                select application_type from {SCHEMA}.licenses
                where license_no = %s
                  and license_start_date <= %s
                  and license_start_date >= %s::date - interval 'P2Y'
                  and license_start_date = (
                      select max(license_start_date) from {SCHEMA}.licenses
                      where license_no = %s
                        and license_start_date <= %s
                        and license_start_date >= %s::date - interval 'P2Y'
                  )
                """,
                (
                    license_no,
                    first_seen,
                    first_seen,
                    license_no,
                    first_seen,
                    first_seen,
                ),
            )
            latest_set = {r[0] for r in cur.fetchall()}
        assert got in latest_set, (
            f"facility {license_no}: as-of pulled {got!r}, "
            f"expected one of {latest_set!r}"
        )


def test_graph_chain_degree_is_causal(food_db):
    """DEGREE over chain_edges equals a manual incident-edge count bounded by
    knowable_at <= as_of — early as-of dates must see fewer or equal edges."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        cur.execute(f"""
            select distinct c.license_no from {cohort} c
            join {SCHEMA}.chain_edges e
              on c.license_no in (e.source_license, e.target_license)
            order by c.license_no limit 5
            """)
        sampled = [r[0] for r in cur.fetchall()]
    assert sampled, "no cohort facility participates in a chain"

    early, late = AS_OF_DATES[0], AS_OF_DATES[-1]
    degree_sql = (
        f"select count(*) from {SCHEMA}.chain_edges "
        "where (source_license = %s or target_license = %s) "
        "and knowable_at <= %s"
    )
    for license_no in sampled:
        for as_of in (early, late):
            got = _value(rows, as_of, license_no, "DEGREE(facilities.chain_edges)")
            want = expect_sql(food_db, degree_sql, (license_no, license_no, as_of))
            if want == 0:
                assert got is None or int(got) == 0
            else:
                assert got is not None and int(got) == want
        early_count = expect_sql(food_db, degree_sql, (license_no, license_no, early))
        late_count = expect_sql(food_db, degree_sql, (license_no, license_no, late))
        assert early_count <= late_count


def test_graph_clique_families_have_closed_form_values(food_db):
    """Chain edges form disjoint cliques, so every link-prediction family has
    a closed-form expectation from the chain size n visible at the as-of date:
    clustering = 1.0, common-neighbours mean = n-2, Jaccard mean = (n-2)/n,
    Adamic-Adar mean = (n-2)/ln(n-1)."""
    rows, cohort = _run_cohort(food_db)
    with food_db.cursor() as cur:
        # Cohort facilities whose visible chain (edges with knowable_at <= MID)
        # has at least 3 members, with the visible size n.
        cur.execute(
            f"""
            select c.license_no, count(*) + 1 as n_visible
            from {cohort} c
            join {SCHEMA}.chain_edges e
              on c.license_no in (e.source_license, e.target_license)
             and e.knowable_at <= %s
            group by c.license_no
            having count(*) >= 2
            order by c.license_no limit 4
            """,
            (MID,),
        )
        sampled = cur.fetchall()
    assert sampled, "no cohort facility is in a visible chain of size >= 3"

    for license_no, n in sampled:
        clustering = _value(
            rows, MID, license_no, "CLUSTERING_COEFF(facilities.chain_edges)"
        )
        common = _value(
            rows, MID, license_no, "COMMON_NEIGHBOURS_MEAN(facilities.chain_edges)"
        )
        jaccard = _value(rows, MID, license_no, "JACCARD_MEAN(facilities.chain_edges)")
        adamic = _value(
            rows, MID, license_no, "ADAMIC_ADAR_MEAN(facilities.chain_edges)"
        )

        assert math.isclose(
            float(clustering), 1.0, rel_tol=1e-9
        ), f"facility {license_no} (clique n={n}): clustering={clustering}"
        assert math.isclose(float(common), n - 2, rel_tol=1e-9)
        assert math.isclose(float(jaccard), (n - 2) / n, rel_tol=1e-9)
        assert math.isclose(float(adamic), (n - 2) / math.log(n - 1), rel_tol=1e-9)


def test_full_table_cheap_aggregations_smoke(food_db):
    """Scale smoke: cheap aggregators over the FULL inspections table (no
    SubqueryAggregator families, no cohort subsetting of children)."""
    config = load_config(
        "food_inspections",
        aggregations=["count", "sum", "mean", "max", "nunique", "recency"],
        intervals=["P1Y"],
    )
    cohort = make_cohort(
        food_db,
        source_table=f"{SCHEMA}.facilities",
        order_by="license_no",
        n=20,
    )
    retarget(config, "facilities", cohort)
    make_as_of_dates(food_db, [MID])
    rows = run_featurizer(food_db, config)
    assert len(rows) == 20
    count_cols = feature_columns(rows, "COUNT(inspections.inspection_id)")
    assert count_cols
    assert any(row[count_cols[0]] is not None for row in rows)
