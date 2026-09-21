"""A paired cohort narrows the child reads too (the #10 follow-up).

``as_of_dates: {id_column: …}`` first narrowed the target's base read only. Every
child was still aggregated for every entity, each date, and the rows of the
entities outside the date's cohort were then thrown away by the join: on
dirtyduck all-agg with two dates, 6.5 of 8 seconds, for a cohort that is 2,184 of
44,338 rows. The reporter of #10 measured the same thing as a cost per date that
grows with the accumulated history.

A faster wrong answer is the failure mode, so narrowing is decided from what the
plan actually reads, and it stays out of the way whenever it could move a value:

- a child is narrowed only when the target's aggregations are its ONLY readers.
  A second parent that aggregates it, or an entity that looks it up, needs rows
  the cohort does not name;
- a window partitions by the child's ``id``. When that is not the join key, the
  rows kept are whole partitions (every row whose id has a row in the cohort),
  never a slice of one;
- a population-level transformer (``avg(x) over ()``) narrows nothing, as before;
- the planner passes (peer groups, spatial, graph) read base tables in CTEs of
  their own and are not reached.

The oracle everywhere is parity with the dense run on the pairs.
"""

from __future__ import annotations

import copy
import os

import pytest

from tests.test_cohort_pairs import (
    CONFIG,
    DATES,
    PAIRS,
    SPINE,
    _featurizer,
    _load_chain,
    _load_flat,
    _paired,
    _population,
    _rows,
)
from tests.utils.render_baseline import chain_config

COHORT = (
    '(select _cohort."cohort_id" from as_of_dates _cohort '
    "where _cohort.as_of_date = aod.as_of_date)"
)


def _synth(query: str, alias: str) -> str:
    start = query.index(f"{alias}_synth as (")
    return query[start : query.index(f"-- transform {alias}", start)]


def _with_child(config: dict, **changes) -> dict:
    config = copy.deepcopy(config)
    child = config["entities"][1]
    for key, value in changes.items():
        if value is None:
            child.pop(key, None)
        else:
            child[key] = value
    return config


# --------------------------------------------------------------- the predicate


def test_a_child_whose_partition_is_the_join_key_is_cut_on_the_key(tmp_path) -> None:
    """``id`` and the foreign key are one column: a window's partition is a key
    group, so keeping a key keeps its partitions whole."""
    config = _with_child(CONFIG, id="customer_id")
    synth = _synth(_featurizer(tmp_path, _paired(config)).query, "orders")

    assert (
        f'where orders."customer_id" in {COHORT} '
        'and orders."ordered_at" <= aod.as_of_date'
    ) in synth
    assert "_rows" not in synth


def test_a_child_without_an_id_has_no_window_and_is_cut_on_the_key(tmp_path) -> None:
    config = _with_child(CONFIG, id=None)
    synth = _synth(_featurizer(tmp_path, _paired(config)).query, "orders")

    assert f'where orders."customer_id" in {COHORT} and ' in synth
    assert "_rows" not in synth


def test_a_child_with_an_id_of_its_own_keeps_whole_partitions(tmp_path) -> None:
    """Windows partition by ``order_id``, and nothing says an id is unique or
    stays under one customer. So the rows kept are every row whose id has a row
    in the cohort."""
    synth = _synth(_featurizer(tmp_path, _paired(CONFIG)).query, "orders")

    assert (
        'where orders."order_id" in (select _rows."order_id" from orders _rows '
        f'where _rows."customer_id" in {COHORT}) '
        'and orders."ordered_at" <= aod.as_of_date'
    ) in synth


NON_ID_KEY = {
    "target": "requests",
    "max_depth": 2,
    "intervals": [],
    "aggregations": ["count", "sum"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "requests",
            "id": "request_id",
            "table": "requests",
            "variables": {"area": {"type": "index"}},
        },
        {
            "alias": "backlog",
            "table": "backlog",
            "temporal_ix": "day",
            "variables": {"open_requests": {"type": "numeric"}},
        },
    ],
    "relationships": [
        {
            "parent": {"entity": "requests", "key": "area"},
            "child": {"entity": "backlog", "key": "area"},
        }
    ],
}


def test_a_join_on_another_target_column_goes_through_the_target(tmp_path) -> None:
    """The pairs name target ids; the child is joined on ``area``. The keys to
    keep are the areas of the cohort's requests."""
    synth = _synth(_featurizer(tmp_path, _paired(NON_ID_KEY)).query, "backlog")

    assert (
        'where backlog."area" in (select _target."area" from requests _target '
        f'where _target."request_id" in {COHORT}) and '
    ) in synth


PARALLEL = {
    "target": "games",
    "max_depth": 2,
    "intervals": [],
    "aggregations": ["count", "sum"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "games",
            "id": "game_id",
            "table": "games",
            "variables": {
                "home_id": {"type": "index"},
                "away_id": {"type": "index"},
            },
        },
        {
            "alias": "team_games",
            "table": "team_games",
            "temporal_ix": "played_on",
            "variables": {"goals": {"type": "numeric"}},
        },
    ],
    "relationships": [
        {
            "name": "home",
            "parent": {"entity": "games", "key": "home_id"},
            "child": {"entity": "team_games", "key": "team_id"},
        },
        {
            "name": "away",
            "parent": {"entity": "games", "key": "away_id"},
            "child": {"entity": "team_games", "key": "team_id"},
        },
    ],
}


def test_two_relationships_to_the_target_keep_the_rows_of_either(tmp_path) -> None:
    synth = _synth(_featurizer(tmp_path, _paired(PARALLEL)).query, "team_games")

    for key in ("home_id", "away_id"):
        assert (
            f'team_games."team_id" in (select _target."{key}" from games _target '
            f'where _target."game_id" in {COHORT})'
        ) in synth
    assert " or " in synth


# ------------------------------------------------------- when nothing is cut


def _no_cut(query: str, alias: str) -> None:
    assert "_cohort" not in _synth(query, alias), alias


def test_a_grandchild_is_read_by_the_child_not_by_the_target(tmp_path) -> None:
    query = _featurizer(tmp_path, _paired(chain_config())).query

    assert f'_rows."store_id" in {COHORT}' in _synth(query, "orders")
    _no_cut(query, "items")


SECOND_PARENT = {
    "target": "customers",
    "max_depth": 3,
    "intervals": [],
    "aggregations": ["count", "sum"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "customers",
            "id": "customer_id",
            "table": "customers",
            "variables": {"age": {"type": "numeric"}},
        },
        {
            "alias": "orders",
            "id": "order_id",
            "table": "orders",
            "temporal_ix": "ordered_at",
            "variables": {"amount": {"type": "numeric"}},
        },
        {"alias": "regions", "id": "region_id", "table": "regions"},
    ],
    "relationships": [
        {
            "parent": {"entity": "customers", "key": "customer_id"},
            "child": {"entity": "orders", "key": "customer_id"},
        },
        {
            "parent": {"entity": "regions", "key": "region_id"},
            "child": {"entity": "orders", "key": "region_id"},
        },
        {
            # Named: the region's order count and the customer's own would
            # otherwise land on the customer under one column name (ADR-0008).
            "name": "region",
            "parent": {"entity": "regions", "key": "region_id"},
            "child": {"entity": "customers", "key": "region_id"},
        },
    ],
}


def test_a_child_that_a_second_parent_aggregates_is_not_cut(tmp_path) -> None:
    """``regions`` sums every order of the region, and the customer receives that
    sum: it needs the orders of customers the cohort does not name."""
    query = _featurizer(tmp_path, _paired(SECOND_PARENT)).query

    assert "orders_aggs_for_regions" in query
    _no_cut(query, "orders")
    _no_cut(query, "regions")


LOOKED_UP = {
    "target": "customers",
    "max_depth": 3,
    "intervals": [],
    "aggregations": ["count", "sum", "max"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "customers",
            "id": "customer_id",
            "table": "customers",
            "variables": {"age": {"type": "numeric"}},
        },
        {
            "alias": "orders",
            "id": "order_id",
            "table": "orders",
            "temporal_ix": "ordered_at",
            "variables": {"amount": {"type": "numeric"}},
        },
        {
            "alias": "plans",
            "id": "plan_id",
            "table": "plans",
            "temporal_ix": "started_on",
            "variables": {"fee": {"type": "numeric"}},
        },
    ],
    "relationships": [
        {
            "parent": {"entity": "customers", "key": "customer_id"},
            "child": {"entity": "orders", "key": "customer_id"},
        },
        {
            "parent": {"entity": "customers", "key": "customer_id"},
            "child": {"entity": "plans", "key": "customer_id"},
        },
        {
            "parent": {"entity": "plans", "key": "plan_id"},
            "child": {"entity": "orders", "key": "plan_id"},
        },
    ],
}


def test_a_child_that_another_entity_looks_up_is_not_cut(tmp_path) -> None:
    """An order pulls its plan's fee, and the plan may belong to a customer the
    cohort does not name. ``orders`` itself is read by the target alone."""
    query = _featurizer(tmp_path, _paired(LOOKED_UP)).query

    _no_cut(query, "plans")
    assert f'_rows."customer_id" in {COHORT}' in _synth(query, "orders")


def test_a_population_level_transformer_cuts_no_read_at_all(tmp_path) -> None:
    query = _featurizer(tmp_path, _paired(_population(CONFIG))).query

    _no_cut(query, "orders")
    _no_cut(query, "customers")
    assert query.count("_cohort.as_of_date = aod.as_of_date") == 1  # the post-filter


def test_without_the_block_no_child_is_cut(tmp_path) -> None:
    for config in (CONFIG, chain_config(), NON_ID_KEY, PARALLEL, LOOKED_UP):
        assert "_cohort" not in _featurizer(tmp_path, config).query


def test_the_output_columns_do_not_depend_on_the_cut(tmp_path) -> None:
    dense = _featurizer(tmp_path, CONFIG)
    paired_dir = tmp_path / "paired"
    paired_dir.mkdir()
    paired = _featurizer(paired_dir, _paired(CONFIG))

    assert [f.column for f in paired.feature_manifest] == [
        f.column for f in dense.feature_manifest
    ]


def test_planning_twice_renders_the_same_query(tmp_path) -> None:
    """The cut is decided from a first traversal and rendered by a second."""
    f = _featurizer(tmp_path, _paired(chain_config()))
    first = f.query
    assert _featurizer(tmp_path, _paired(chain_config())).query == first
    assert first.count(SPINE) == 1


def test_every_column_group_and_the_preamble_carry_the_childs_cut(tmp_path) -> None:
    f = _featurizer(tmp_path, _paired(chain_config()), materialize_threshold=1)
    assert f._grouped().materialization is not None, "expected a preamble"
    ddl = "\n".join(f.materialization_ddl)

    assert f'_rows."store_id" in {COHORT}' in ddl
    assert "from (select distinct as_of_date from as_of_dates) aod" in ddl


# ------------------------------------------------------------------ execution


def _load_non_id_key(cur) -> None:
    cur.execute("create temp table requests (request_id int, area int)")
    cur.execute("insert into requests values (1, 10), (2, 10), (3, 20), (4, 30)")
    cur.execute("create temp table backlog (area int, day date, open_requests int)")
    cur.execute(
        "insert into backlog values "
        "(10, '2024-01-10', 5), (10, '2024-02-15', 7), (10, '2024-03-20', 11), "
        "(20, '2024-01-05', 1), (20, '2024-03-05', 3), "
        "(30, '2024-02-25', 40), (30, '2024-05-01', 2)"
    )


def _load_parallel(cur) -> None:
    cur.execute("create temp table games (game_id int, home_id int, away_id int)")
    cur.execute(
        "insert into games values (1, 100, 200), (2, 200, 300), "
        "(3, 300, 100), (4, 400, 200)"
    )
    cur.execute("create temp table team_games (team_id int, played_on date, goals int)")
    cur.execute(
        "insert into team_games values "
        "(100, '2024-01-10', 3), (100, '2024-02-20', 1), (100, '2024-03-10', 4), "
        "(200, '2024-01-15', 2), (200, '2024-03-25', 6), "
        "(300, '2024-02-05', 5), (300, '2024-03-30', 7), "
        "(400, '2024-01-01', 9), (400, '2024-05-01', 8)"
    )


def _load_second_parent(cur) -> None:
    cur.execute("create temp table regions (region_id int)")
    cur.execute("insert into regions values (7), (8)")
    cur.execute(
        "create temp table customers (customer_id int, age numeric, region_id int)"
    )
    cur.execute(
        "insert into customers values (1, 30, 7), (2, 40, 7), (3, 50, 8), (4, 60, 8)"
    )
    cur.execute(
        "create temp table orders (order_id int, customer_id int, region_id int, "
        "ordered_at date, amount numeric)"
    )
    cur.execute(
        "insert into orders values "
        "(1, 1, 7, '2024-01-10', 10), (2, 1, 7, '2024-03-15', 20), "
        "(3, 2, 7, '2024-01-20', 5), (4, 2, 7, '2024-02-20', 7), "
        "(5, 3, 8, '2024-02-25', 100), (6, 4, 8, '2024-01-05', 1)"
    )


def _load_looked_up(cur) -> None:
    cur.execute("create temp table customers (customer_id int, age numeric)")
    cur.execute("insert into customers values (1, 30), (2, 40), (3, 50), (4, 60)")
    cur.execute(
        "create temp table plans (plan_id int, customer_id int, "
        "started_on date, fee numeric)"
    )
    cur.execute(
        "insert into plans values (11, 1, '2024-01-01', 9), (12, 2, '2024-01-01', 19), "
        "(13, 3, '2024-01-01', 29), (14, 4, '2024-01-01', 39)"
    )
    # Orders 2 and 5 are on a plan that belongs to ANOTHER customer.
    cur.execute(
        "create temp table orders (order_id int, customer_id int, plan_id int, "
        "ordered_at date, amount numeric)"
    )
    cur.execute(
        "insert into orders values "
        "(1, 1, 11, '2024-01-10', 10), (2, 1, 14, '2024-03-15', 20), "
        "(3, 2, 12, '2024-01-20', 5), (4, 2, 12, '2024-02-20', 7), "
        "(5, 3, 12, '2024-02-25', 100), (6, 4, 14, '2024-01-05', 1)"
    )


# A window that crosses customers: ``track`` 1 holds the orders of customers 1
# and 2, so ``lag_1`` of customer 1's March order is customer 2's February one.
# Cutting on the key alone drops that row whenever customer 2 is not in the
# date's cohort, and the lag turns into customer 1's own January order.
CROSSING = {
    **copy.deepcopy(CONFIG),
    "intervals": [],
    "aggregations": ["sum", "max"],
    "transformations": ["identity", "lag_1"],
}
CROSSING["entities"][1]["id"] = "track"


def _load_crossing(cur) -> None:
    cur.execute("create temp table customers (customer_id int, age numeric)")
    cur.execute("insert into customers values (1, 30), (2, 40), (3, 50), (4, 60)")
    cur.execute(
        "create temp table orders (track int, customer_id int, "
        "ordered_at date, amount numeric)"
    )
    cur.execute(
        "insert into orders values "
        "(1, 1, '2024-01-10', 10), (1, 2, '2024-02-20', 7), (1, 1, '2024-03-15', 20), "
        "(1, 2, '2024-03-25', 9), (2, 3, '2024-01-25', 100), (2, 4, '2024-02-05', 1), "
        "(2, 3, '2024-02-25', 50), (2, 4, '2024-03-05', 2)"
    )


def _asof_lookup() -> dict:
    """The child receives an as-of lookup; the lookup table is nobody's child."""
    config = copy.deepcopy(CONFIG)
    config["intervals"] = []
    config["aggregations"] = ["count", "sum", "max"]
    config["entities"].append(
        {
            "alias": "rates",
            "id": "rate_id",
            "table": "rates",
            "temporal_ix": "valid_from",
            "variables": {"rate": {"type": "numeric"}},
        }
    )
    config["entities"][1]["variables"]["zone"] = {"type": "index"}
    config["relationships"].append(
        {
            "parent": {"entity": "rates", "key": "zone"},
            "child": {"entity": "orders", "key": "zone"},
            "temporal": {"mode": "as_of"},
        }
    )
    return config


def _load_asof_lookup(cur) -> None:
    cur.execute("create temp table customers (customer_id int, age numeric)")
    cur.execute("insert into customers values (1, 30), (2, 40), (3, 50), (4, 60)")
    cur.execute(
        "create temp table rates (rate_id int, zone int, valid_from date, rate numeric)"
    )
    cur.execute(
        "insert into rates values (1, 1, '2024-01-01', 0.1), (2, 1, '2024-03-01', 0.2), "
        "(3, 2, '2024-01-01', 0.5), (4, 2, '2024-02-22', 0.7)"
    )
    cur.execute(
        "create temp table orders (order_id int, customer_id int, zone int, "
        "ordered_at date, amount numeric)"
    )
    cur.execute(
        "insert into orders values "
        "(1, 1, 1, '2024-01-10', 10), (2, 1, 2, '2024-03-15', 20), "
        "(3, 2, 1, '2024-01-20', 5), (4, 2, 1, '2024-02-20', 7), "
        "(5, 2, 2, '2024-03-25', 9), (6, 3, 2, '2024-02-25', 100), "
        "(7, 4, 1, '2024-01-05', 1)"
    )


def _dated_target() -> dict:
    """ADR-0017's cut and the pairing meet on the target; the child is cut too."""
    config = copy.deepcopy(CONFIG)
    config["entities"][0]["temporal_ix"] = "signed_up"
    return config


def _load_dated_target(cur) -> None:
    _load_flat(cur)
    cur.execute("alter table customers add column signed_up date")
    cur.execute(
        "update customers set signed_up = case customer_id "
        "when 1 then date '2024-01-01' when 2 then date '2024-01-15' "
        "when 3 then date '2024-02-10' else date '2024-03-20' end"
    )


SHAPES = {
    "whole-partitions": (CONFIG, _load_flat, "customer_id", {}, "orders"),
    "key-is-the-partition": (
        _with_child(CONFIG, id="customer_id"),
        _load_flat,
        "customer_id",
        {},
        "orders",
    ),
    "child-without-id": (
        _with_child(CONFIG, id=None),
        _load_flat,
        "customer_id",
        {},
        "orders",
    ),
    "non-id-key": (NON_ID_KEY, _load_non_id_key, "request_id", {}, "backlog"),
    "parallel-relationships": (PARALLEL, _load_parallel, "game_id", {}, "team_games"),
    "grandchild": (chain_config(), _load_chain, "store_id", {}, "orders"),
    "temp-table-preamble": (
        chain_config(),
        _load_chain,
        "store_id",
        {"materialize_threshold": 1},
        "orders",
    ),
    "second-parent": (SECOND_PARENT, _load_second_parent, "customer_id", {}, None),
    "looked-up-child": (LOOKED_UP, _load_looked_up, "customer_id", {}, "orders"),
    "window-crossing-the-key": (CROSSING, _load_crossing, "customer_id", {}, "orders"),
    "as-of-lookup-on-the-child": (
        _asof_lookup(),
        _load_asof_lookup,
        "customer_id",
        {},
        "orders",
    ),
    "dated-target": (_dated_target(), _load_dated_target, "customer_id", {}, "orders"),
}


def _features_only(frame):
    """Without the dated target's own temporal index, which is not a number."""
    return frame.drop(columns=[c for c in frame.columns if c == "signed_up"])


def _dense_and_paired(tmp_path, config, load, id_name, kwargs):
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    dense_f = _featurizer(tmp_path, config, **kwargs)
    paired_dir = tmp_path / "paired"
    paired_dir.mkdir(exist_ok=True)
    paired_f = _featurizer(paired_dir, _paired(config), **kwargs)

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            load(cur)
            cur.execute("create temp table as_of_dates (as_of_date date)")
            cur.executemany(
                "insert into as_of_dates values (%s)", [(d,) for d in DATES]
            )
        dense = _rows(_features_only(dense_f.to_dataframe(connection=conn)), id_name)
        conn.rollback()

        with conn.cursor() as cur:
            load(cur)
            cur.execute(
                "create temp table as_of_dates (as_of_date date, cohort_id int)"
            )
            cur.executemany("insert into as_of_dates values (%s, %s)", PAIRS)
        paired = _rows(_features_only(paired_f.to_dataframe(connection=conn)), id_name)
        conn.rollback()
    return paired_f, dense, paired


@pytest.mark.integration
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_narrowed_child_gives_the_dense_values_on_the_pairs(tmp_path, shape) -> None:
    config, load, id_name, kwargs, narrowed = SHAPES[shape]
    paired_f, dense, paired = _dense_and_paired(tmp_path, config, load, id_name, kwargs)

    # What ran is what this file is about: the child's read carries the cut
    # (or, for the second parent, provably does not).
    sql = "\n".join(list(paired_f.query_groups.values()) + paired_f.materialization_ddl)
    cuts = sql.count("_cohort.as_of_date = aod.as_of_date")
    target_cuts = sql.count(f'{config["target"]}."{id_name}" in (select _cohort.')
    assert target_cuts >= 1
    if narrowed is None:
        assert cuts == target_cuts
    else:
        assert cuts > target_cuts, f"{narrowed} was not narrowed"

    # ADR-0017: a dated target emits a pair only once its row exists.
    expected = {pair: dense[pair] for pair in PAIRS if pair in dense}
    assert expected, "no pair survived; the shape checks nothing"
    assert paired == expected
    # The oracle must be able to fail: the values differ across the pairs.
    assert len(set(paired.values())) > 1


@pytest.mark.integration
def test_cutting_a_crossing_window_on_the_key_alone_would_move_a_value(
    tmp_path, monkeypatch
) -> None:
    """The reason for whole partitions, shown by taking them away."""
    from featurizer.planner import FeaturePlanner

    monkeypatch.setattr(
        FeaturePlanner, "_partition_is_the_key", lambda self, entity, rels: True
    )
    config, load, id_name, kwargs, _ = SHAPES["window-crossing-the-key"]
    _, dense, paired = _dense_and_paired(tmp_path, config, load, id_name, kwargs)

    assert paired != {pair: dense[pair] for pair in PAIRS}


# ------------------------------------------------ planner passes on the child


def _passes_on_the_child() -> dict:
    """``owners <- series <- events``, with the peer-group, spatial and
    graph-relationship passes on ``series``: the child the cohort narrows. Its
    peers and its graph neighbours belong to OTHER owners, so a cut that reached
    a pass would change what the owner receives."""
    from tests.integration.test_future_row_planner_passes import _passes_config

    config = copy.deepcopy(_passes_config())
    config["target"] = "owners"
    config["max_depth"] = 3
    config["entities"].insert(
        0, {"alias": "owners", "table": "owners", "id": "owner_id"}
    )
    config["relationships"].append(
        {
            "parent": {"entity": "owners", "key": "owner_id"},
            "child": {"entity": "series", "key": "owner_ref"},
        }
    )
    return config


@pytest.mark.integration
def test_the_planner_passes_on_a_narrowed_child_give_the_dense_values(tmp_path) -> None:
    import psycopg

    from tests.integration.test_future_row_planner_passes import AS_OF, TABLES

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    config = _passes_on_the_child()
    pairs = [(AS_OF[0], 1), (AS_OF[1], 2), (AS_OF[1], 1)]

    def load(cur) -> None:
        for table, columns, knowable, _ in TABLES:
            cur.execute(f"create temp table {table} ({columns})")
            cur.execute(f"insert into {table} values {knowable}")
        cur.execute("alter table series add column owner_ref int")
        cur.execute(
            "update series set owner_ref = case when series_id <= 2 then 1 else 2 end"
        )
        cur.execute("create temp table owners (owner_id int)")
        cur.execute("insert into owners values (1), (2)")

    dense_f = _featurizer(tmp_path, config)
    paired_dir = tmp_path / "paired"
    paired_dir.mkdir()
    paired_f = _featurizer(paired_dir, _paired(config))
    assert "_cohort" in _synth(paired_f.query, "series")
    for family in (
        "PEER_",
        "COLOCATION_COUNT",
        "KDE_INTENSITY",
        "DEGREE",
        "NEIGHBOUR_",
    ):
        assert any(family in f.column for f in dense_f.feature_manifest), family

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            load(cur)
            cur.execute("create temp table as_of_dates (as_of_date date)")
            cur.executemany(
                "insert into as_of_dates values (%s)", [(d,) for d in AS_OF]
            )
        dense = _rows(dense_f.to_dataframe(connection=conn), "owner_id")
        conn.rollback()
        with conn.cursor() as cur:
            load(cur)
            cur.execute(
                "create temp table as_of_dates (as_of_date date, cohort_id int)"
            )
            cur.executemany("insert into as_of_dates values (%s, %s)", pairs)
        paired = _rows(paired_f.to_dataframe(connection=conn), "owner_id")
        conn.rollback()

    expected = {(str(d), owner): dense[(str(d), owner)] for d, owner in pairs}
    assert paired == expected
    assert len(set(paired.values())) > 1
