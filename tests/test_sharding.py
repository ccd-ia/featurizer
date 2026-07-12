"""DB-free shape tests for column-group sharding (issue #7).

These assert on the *structure* of the sharded queries — group count, per-group
column budgets, feature coverage (no feature dropped or duplicated), and the
public API contract — without executing any SQL. Execution + value-equivalence
on real PostgreSQL lives in ``tests/integration/test_sharding.py``.
"""

from __future__ import annotations

import re
import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import (
    PG_MAX_TABLE_COLUMNS,
    PG_MAX_TARGET_LIST,
    ColumnGroupSharder,
    MaterializationPlanner,
)


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path, validate=False)


def _wide_config(
    n_vars: int = 12,
    intervals: list[str] | None = None,
    aggregations: list[str] | None = None,
    transformations: list[str] | None = None,
) -> dict:
    """A customers<-orders config wide enough to exceed the 1664 limit."""
    variables = {f"v{i}": {"type": "numeric"} for i in range(n_vars)}
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": intervals or ["P1W", "P1M", "P3M", "P6M", "P1Y", "P2Y"],
        "aggregations": aggregations or ["count", "sum", "mean", "min", "max"],
        "transformations": transformations or ["identity", "abs", "sqrt"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": variables,
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _narrow_config() -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1M"],
        "aggregations": ["count", "sum", "mean"],
        "transformations": ["identity", "abs"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _select_list_columns(cte_select: str) -> int:
    """Count top-level columns in a select list rendered by the sharder.

    The sharder joins projections with ``,\\n        `` and column expressions
    do not themselves contain that exact separator, so counting it +1 is a
    faithful column count for the assertions here.
    """
    return cte_select.count(",\n        ") + 1


def _cte_select_lists(sql: str) -> dict[str, str]:
    """Extract the select list of every ``<name> as ( select <list> from ...``."""
    out: dict[str, str] = {}
    for m in re.finditer(r"(\w+) as \(\s*select\s*(.*?)\n\s*from ", sql, re.S):
        out[m.group(1)] = m.group(2)
    return out


# ------------------------------------------------------------------ #
# Narrow config: single group, API parity with .query
# ------------------------------------------------------------------ #


def test_narrow_config_fits_single_group():
    f = _featurizer(_narrow_config())
    sharder = ColumnGroupSharder(f._plan)
    assert sharder.fits_single_group is True
    groups = f.query_groups
    assert list(groups) == ["group_000"]


def test_narrow_query_groups_equals_query():
    f = _featurizer(_narrow_config())
    groups = f.query_groups
    # A single-group config returns exactly the single-query SQL.
    assert groups["group_000"] == f.query


def test_narrow_query_does_not_raise():
    f = _featurizer(_narrow_config())
    # Fits in one valid query; .query returns SQL rather than raising.
    assert "customers_transform" in f.query


# ------------------------------------------------------------------ #
# Wide config: many groups, every CTE under the limit, full coverage
# ------------------------------------------------------------------ #


def test_wide_config_partitions_into_multiple_groups():
    f = _featurizer(_wide_config())
    sharder = ColumnGroupSharder(f._plan)
    assert sharder.fits_single_group is False
    assert sharder.n_groups > 1
    assert len(f.query_groups) == sharder.n_groups


def test_wide_query_raises_pointing_at_sharded_api():
    f = _featurizer(_wide_config())
    with pytest.raises(ValueError, match="too wide"):
        _ = f.query
    # The error names the sharded escape hatches.
    try:
        _ = f.query
    except ValueError as exc:
        assert "query_groups" in str(exc)
        assert "to_parquet" in str(exc)


def test_every_group_cte_under_postgres_limit():
    f = _featurizer(_wide_config())
    for gid, sql in f.query_groups.items():
        for cte_name, select_list in _cte_select_lists(sql).items():
            n = _select_list_columns(select_list)
            assert n <= PG_MAX_TABLE_COLUMNS, (
                f"{gid}/{cte_name} projects {n} columns, over the "
                f"{PG_MAX_TABLE_COLUMNS}-column table limit"
            )
            assert n <= PG_MAX_TARGET_LIST


def test_feature_columns_partition_exactly():
    """Union of group feature columns == full set; no drop, no dup (except keys)."""
    f = _featurizer(_wide_config())
    full = [c.name for c in f._plan.cte_specs["customers_transform"].columns]
    full_set = set(full)
    assert len(full) == len(full_set), "planner produced duplicate feature names"

    seen: set[str] = set()
    transform_name = "customers_transform"
    for sql in f.query_groups.values():
        select_lists = _cte_select_lists(sql)
        # The target transform CTE's projected feature names for this group.
        select = select_lists[transform_name]
        # Each transform projection is ``<expr> as "<name>"`` or a pass-through;
        # match the quoted output identifier.
        names = set(re.findall(r'as ("(?:[^"]|"")*")', select))
        # Drop the identifier (id) columns which carry into every group.
        feature_names = {n for n in names if n in full_set}
        overlap = seen & feature_names
        assert not overlap, f"feature columns duplicated across groups: {overlap}"
        seen |= feature_names

    # Every feature appears in exactly one group.
    assert seen == full_set, f"missing from groups: {full_set - seen}"


def test_groups_lead_with_join_keys():
    f = _featurizer(_wide_config())
    for sql in f.query_groups.values():
        # Outer select leads with as_of_date then t.* over the transform CTE,
        # whose first projected column is the target id.
        assert "select aod.as_of_date, t.*" in sql
        transform = _cte_select_lists(sql)["customers_transform"]
        first_col = transform.strip().split(",")[0].strip()
        assert first_col == "customer_id"


# ------------------------------------------------------------------ #
# Partition-size knob and fan-out guard
# ------------------------------------------------------------------ #


def test_smaller_group_size_makes_more_groups():
    f = _featurizer(_wide_config())
    big = ColumnGroupSharder(f._plan, max_columns_per_group=1400).n_groups
    small = ColumnGroupSharder(f._plan, max_columns_per_group=200).n_groups
    assert small > big


def test_zero_or_negative_group_size_rejected():
    f = _featurizer(_narrow_config())
    with pytest.raises(ValueError, match="positive integer"):
        ColumnGroupSharder(f._plan, max_columns_per_group=0)


def test_key_columns_are_as_of_date_and_id():
    f = _featurizer(_wide_config())
    built = ColumnGroupSharder(f._plan).build()
    assert built.key_columns == ["as_of_date", "customer_id"]


# ------------------------------------------------------------------ #
# Default partition size respects the PostgreSQL table-column limit
# ------------------------------------------------------------------ #


def test_default_group_size_under_table_limit():
    from featurizer.sharding import DEFAULT_MAX_COLUMNS_PER_GROUP

    # The default leaves headroom under the 1600-column table limit for the
    # carried key columns.
    assert DEFAULT_MAX_COLUMNS_PER_GROUP < PG_MAX_TABLE_COLUMNS


# ------------------------------------------------------------------ #
# Deep chains: only target-level aggs are pruned; deeper aggs stay whole
# ------------------------------------------------------------------ #


def _depth3_config() -> dict:
    """stores <- orders <- items: a depth-3 chain (target = stores)."""
    return {
        "target": "stores",
        "max_depth": 3,
        "intervals": [],
        "aggregations": ["count", "sum", "mean"],
        "transformations": ["identity", "abs"],
        "entities": [
            {"alias": "stores", "table": "stores", "id": "store_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {
                    "store_id": {"type": "index"},
                    "total": {"type": "numeric"},
                },
            },
            {
                "alias": "items",
                "table": "items",
                "id": "item_id",
                "temporal_ix": "added_at",
                "variables": {
                    "order_id": {"type": "index"},
                    "price": {"type": "numeric"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "stores", "key": "store_id"},
                "child": {"entity": "orders", "key": "store_id"},
            },
            {
                "parent": {"entity": "orders", "key": "order_id"},
                "child": {"entity": "items", "key": "order_id"},
            },
        ],
    }


def test_only_target_level_aggs_are_pruned():
    """A deeper-chain agg CTE (items->orders) is NOT treated as prunable.

    Pruning a non-target agg against the *target's* synth columns would empty
    its select list; it must be emitted whole. Only the target-level agg
    (orders->stores) is prunable.
    """
    f = _featurizer(_depth3_config())
    sharder = ColumnGroupSharder(f._plan)
    assert sharder._target_agg_ctes == {"orders_aggs_for_stores"}
    assert "items_aggs_for_orders" not in sharder._target_agg_ctes


def test_depth3_groups_cover_all_features_and_stay_under_limit():
    f = _featurizer(_depth3_config())
    # Force several groups with a tiny budget so the deeper-agg path is exercised.
    built = ColumnGroupSharder(f._plan, max_columns_per_group=3).build()
    assert len(built.queries) > 1

    full = {c.name for c in f._plan.cte_specs["stores_transform"].columns}
    items_agg_cols = {
        c.name for c in f._plan.cte_specs["items_aggs_for_orders"].columns
    }
    seen: set[str] = set()
    for sql in built.queries.values():
        # The deeper-chain agg, when present in a group, is emitted WHOLE — every
        # one of its column names appears (it is not pruned against the target's
        # synth columns, which would empty it). It is rendered from the planner's
        # verbatim string, so check by name presence rather than separator count.
        if "items_aggs_for_orders as (" in sql:
            for col_name in items_agg_cols:
                assert col_name in sql, f"deeper agg dropped {col_name}"
        select_lists = _cte_select_lists(sql)
        names = set(
            re.findall(r'as ("(?:[^"]|"")*")', select_lists["stores_transform"])
        )
        seen |= {n for n in names if n in full}
    assert seen == full


# ------------------------------------------------------------------ #
# The #7 residual limitation: an oversized NON-TARGET child CTE.
# Sharding the target cannot shrink it (it is reused whole across groups);
# these tests pin the *detection* (which the temp-table materialization layer
# consumes) and the fact that the cascade is inherent: an oversized child agg
# forces its consumer synth/transform over the limit too.
# ------------------------------------------------------------------ #


def _oversized_child_config() -> dict:
    """stores <- orders <- items, with ``items`` wide enough that
    ``items_aggs_for_orders`` alone exceeds the 1664 target-list limit.

    150 numeric item variables × a 16-aggregation set (no intervals → one
    all-time window) ≈ 2 400 aggregate columns, comfortably over 1664.
    """
    item_vars = {"order_id": {"type": "index"}}
    item_vars.update({f"m{i}": {"type": "numeric"} for i in range(150)})
    return {
        "target": "stores",
        "max_depth": 3,
        "intervals": [],
        "aggregations": [
            "count",
            "sum",
            "mean",
            "min",
            "max",
            "stddev",
            "variance",
            "median",
            "nunique",
            "p25",
            "p75",
            "p90",
            "p95",
            "p99",
            "iqr",
            "range",
        ],
        "transformations": ["identity"],
        "entities": [
            {"alias": "stores", "table": "stores", "id": "store_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {
                    "store_id": {"type": "index"},
                    "total": {"type": "numeric"},
                },
            },
            {
                "alias": "items",
                "table": "items",
                "id": "item_id",
                "temporal_ix": "added_at",
                "variables": item_vars,
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "stores", "key": "store_id"},
                "child": {"entity": "orders", "key": "store_id"},
            },
            {
                "parent": {"entity": "orders", "key": "order_id"},
                "child": {"entity": "items", "key": "order_id"},
            },
        ],
    }


def _cte_width(plan, name: str) -> int:
    spec = plan.cte_specs[name]
    return len(spec.key_columns) + len(spec.columns)


def test_oversized_child_agg_is_detected():
    """``items_aggs_for_orders`` exceeds the hard limit and is flagged as an
    intermediate CTE that sharding cannot shrink."""
    f = _featurizer(_oversized_child_config())
    sharder = ColumnGroupSharder(f._plan)
    oversized = sharder._oversized_intermediate_ctes()
    assert "items_aggs_for_orders" in oversized
    assert oversized["items_aggs_for_orders"] > PG_MAX_TARGET_LIST
    # It is a non-target (deeper-chain) agg, so it is NOT prunable per group.
    assert "items_aggs_for_orders" not in sharder._target_agg_ctes


def test_oversized_child_cascade_is_inherent():
    """An oversized child agg forces its consumer synth/transform over the limit
    too — they project its columns (planner.py records every agg column as a
    synth column). So the materialization layer must handle the whole non-target
    chain, not just the leaf agg."""
    f = _featurizer(_oversized_child_config())
    sharder = ColumnGroupSharder(f._plan)
    oversized = sharder._oversized_intermediate_ctes()
    # The whole non-target chain is over the limit.
    assert "orders_synth" in oversized, "consumer synth should cascade over-limit"
    assert "orders_transform" in oversized, "consumer transform should cascade too"
    # Sanity: the synth is at least as wide as the agg it projects.
    assert _cte_width(f._plan, "orders_synth") >= _cte_width(
        f._plan, "items_aggs_for_orders"
    )


def test_oversized_child_excludes_target_level_ctes():
    """The target's own synth/transform and its target-level aggs are pruned per
    group, so they are NOT reported as un-shrinkable oversized intermediates."""
    f = _featurizer(_oversized_child_config())
    sharder = ColumnGroupSharder(f._plan)
    oversized = sharder._oversized_intermediate_ctes()
    assert "stores_synth" not in oversized
    assert "stores_transform" not in oversized
    assert "orders_aggs_for_stores" not in oversized


def test_oversized_child_does_not_fit_single_group():
    f = _featurizer(_oversized_child_config())
    sharder = ColumnGroupSharder(f._plan)
    assert sharder.fits_single_group is False


def test_materialization_key_recorded_for_agg_cte():
    """The planner records the join geometry the temp-table materializer needs:
    the agg CTE's group/join key and the consumer's LEFT JOIN clause."""
    f = _featurizer(_oversized_child_config())
    keys = f._plan.materialization_keys
    assert "items_aggs_for_orders" in keys
    mk = keys["items_aggs_for_orders"]
    assert mk.join_key == "order_id"
    # The join clause names the CTE (the materializer swaps it per shard).
    assert "items_aggs_for_orders" in mk.join_statement
    assert "order_id" in mk.join_statement


def test_oversized_child_warn_oversized_silent_when_materializable():
    """``warn_oversized`` no longer warns for an oversized child CTE that has a
    join key: it is handled by temp-table materialization (issue #7). The warning
    is reserved for CTEs that cannot be materialized (no id to re-join on).

    loguru holds the import-time ``sys.stderr`` reference, so ``capsys`` cannot
    see its output; add a temporary loguru sink to capture the warnings instead.
    """
    from loguru import logger

    f = _featurizer(_oversized_child_config())
    sharder = ColumnGroupSharder(f._plan)
    # The whole chain is materializable (every entity has an id), so no warning.
    assert set(sharder._oversized_intermediate_ctes()) & _CHAIN
    assert all(name in f._plan.materialization_keys for name in _CHAIN)

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        sharder.warn_oversized()
    finally:
        logger.remove(sink_id)

    assert messages == [], "materializable CTEs should not warn"


# ------------------------------------------------------------------ #
# MaterializationPlanner: temp-table shards for the oversized child chain.
# These assert on the *shape* of the CREATE TEMP TABLE preamble (DB-free);
# execution + value-equivalence on real PostgreSQL lives in the integration
# suite.
# ------------------------------------------------------------------ #

_CHAIN = {"items_aggs_for_orders", "orders_synth", "orders_transform"}


def _materialization_plan():
    f = _featurizer(_oversized_child_config())
    mp = MaterializationPlanner(f._plan)
    return f, mp, mp.build()


def test_materialization_detects_oversized_chain():
    f, mp, _ = _materialization_plan()
    assert set(mp.oversized_ctes()) == _CHAIN
    # Target-level CTEs are pruned per group, never materialized.
    assert "stores_synth" not in mp.oversized_ctes()
    assert "orders_aggs_for_stores" not in mp.oversized_ctes()


def test_materialization_order_is_bottom_up():
    _, mp, _ = _materialization_plan()
    order = mp.materialization_order()
    assert (
        order.index("items_aggs_for_orders")
        < order.index("orders_synth")
        < order.index("orders_transform")
    )


def test_materialization_ddl_are_temp_tables():
    _, _, plan = _materialization_plan()
    assert plan.materialized_ctes == _CHAIN
    assert plan.ddl, "expected a non-empty preamble"
    # The preamble pairs an idempotent `drop … if exists` with each create.
    creates = [d for d in plan.ddl if d.startswith("create temp table __fz_")]
    drops = [d for d in plan.ddl if d.startswith("drop table if exists __fz_")]
    assert creates
    assert len(drops) == len(creates)
    for ddl in creates:
        assert "on commit drop as" in ddl


def test_materialization_shards_stay_under_budget():
    _, mp, plan = _materialization_plan()
    # items_aggs_for_orders (2107 cols) splits into >1 shard at the 1400 budget.
    assert len(plan.shards_by_cte["items_aggs_for_orders"]) > 1
    for shards in plan.shards_by_cte.values():
        for shard in shards:
            assert len(shard.columns) <= mp.max_columns_per_shard


def test_materialization_shards_cover_every_column_once():
    f, _, plan = _materialization_plan()
    for cte_name, shards in plan.shards_by_cte.items():
        full = [c.name for c in f._plan.cte_specs[cte_name].columns]
        seen: list[str] = []
        for shard in shards:
            seen.extend(shard.columns)
        assert seen == full, f"{cte_name}: shards do not partition columns exactly"


def test_items_aggs_ddl_inlines_bounded_upstreams():
    """The leaf agg's source ``items_transform`` (+ ``items_synth``) is bounded,
    so it is pulled into an inline ``with`` rather than materialized."""
    _, _, plan = _materialization_plan()
    agg_ddl = "\n".join(
        s.create_sql for s in plan.shards_by_cte["items_aggs_for_orders"]
    )
    assert "items_synth as (" in agg_ddl
    assert "items_transform as (" in agg_ddl


def test_orders_synth_ddl_joins_items_aggs_temp_shards():
    """``orders_synth`` no longer inlines the oversized child agg; it left-joins
    its temp shards instead."""
    _, _, plan = _materialization_plan()
    synth_ddl = "\n".join(s.create_sql for s in plan.shards_by_cte["orders_synth"])
    assert "items_aggs_for_orders as (" not in synth_ddl
    assert "__fz_items_aggs_for_orders__s000" in synth_ddl


def test_orders_transform_ddl_rejoins_synth_shards():
    """``orders_transform`` reads its synth from the re-joined synth shards.

    The chain is as-of-keyed (``items`` has a temporal index, so the agg carries a
    causal filter), so the shards re-join on ``(as_of_date, order_id)`` and the
    transform carries ``as_of_date`` through."""
    _, _, plan = _materialization_plan()
    tf_ddl = "\n".join(s.create_sql for s in plan.shards_by_cte["orders_transform"])
    assert "__fz_orders_synth__s000" in tf_ddl
    assert "using (as_of_date, order_id)" in tf_ddl


def test_materialization_threshold_knob_forces_small_config():
    """A tiny threshold materializes even a small config's child chain, so the
    DB-free path is exercisable without a genuinely 1664-wide CTE."""
    f = _featurizer(_depth3_config())
    mp = MaterializationPlanner(f._plan, materialize_threshold=1)
    oversized = set(mp.oversized_ctes())
    assert "items_aggs_for_orders" in oversized
    assert "orders_synth" in oversized
    # Target CTEs stay out even at threshold 1 (they are prunable).
    assert "stores_synth" not in oversized


# ------------------------------------------------------------------ #
# ColumnGroupSharder integration: group queries read the materialized
# temp tables; the materialized chain (and its dead upstreams) are dropped
# from the group WITH lists; the preamble rides on GroupedQueries.
# ------------------------------------------------------------------ #


def test_non_oversized_config_has_no_materialization():
    """The common case is unchanged: no preamble, byte-identical group SQL."""
    f = _featurizer(_wide_config())
    built = ColumnGroupSharder(f._plan).build()
    assert built.materialization is None


def test_oversized_child_config_materializes_chain():
    f = _featurizer(_oversized_child_config())
    mplan = ColumnGroupSharder(f._plan).materialization()
    assert mplan is not None
    assert _CHAIN <= mplan.materialized_ctes
    assert mplan.ddl  # a non-empty CREATE TEMP TABLE preamble


def _materialized_depth3_build():
    """Force the small depth-3 chain to materialize via a tiny threshold."""
    f = _featurizer(_depth3_config())
    sharder = ColumnGroupSharder(f._plan, materialize_threshold=1)
    return f, sharder, sharder.build()


def test_materialized_groups_carry_preamble():
    _, _, built = _materialized_depth3_build()
    assert built.materialization is not None
    assert built.materialization.ddl


def test_materialized_groups_read_temp_tables_not_inline_chain():
    _, _, built = _materialized_depth3_build()
    all_sql = "\n".join(built.queries.values())
    # The target-level agg now reads the materialized orders_transform shards.
    assert "__fz_orders_transform__s000" in all_sql
    # None of the materialized chain is emitted as an inline CTE in the groups…
    for cte in (
        "orders_synth as (",
        "orders_transform as (",
        "items_aggs_for_orders as (",
    ):
        assert cte not in all_sql, f"{cte!r} should be a temp table, not inline"
    # …nor are upstreams reachable only through a materialized CTE (dead weight).
    for cte in ("items_synth as (", "items_transform as ("):
        assert cte not in all_sql, f"dead upstream {cte!r} should be dropped"


def test_materialized_groups_stay_under_limit():
    _, _, built = _materialized_depth3_build()
    for sql in built.queries.values():
        for name, select_list in _cte_select_lists(sql).items():
            assert _select_list_columns(select_list) <= PG_MAX_TARGET_LIST


def test_materialized_groups_still_lead_with_join_keys():
    """The (as_of_date, target id) re-join contract survives materialization."""
    _, _, built = _materialized_depth3_build()
    assert built.key_columns == ["as_of_date", "store_id"]
    for sql in built.queries.values():
        assert "select aod.as_of_date, t.*" in sql


# --------------------------------------------------------------------------- #
# Plan-size guardrail (pre-flight PG planner-blowup prediction)
# --------------------------------------------------------------------------- #


def test_plan_size_report_covers_every_group_and_counts_real_closures():
    """One closure count per group, and each count matches the CTEs the
    rendered group query actually carries (`with … as (` occurrences plus the
    target synth/transform are the closure by construction)."""
    f = _featurizer(_wide_config())
    sharder = ColumnGroupSharder(f._plan)
    report = sharder.plan_size_report()
    built = sharder.build()
    assert list(report.keys()) == list(built.queries.keys())
    for gid, sql in built.queries.items():
        assert report[gid] == len(re.findall(r"\bas \(", sql)), gid
        assert report[gid] >= 2  # target synth + transform at minimum


def test_warn_plan_size_silent_under_threshold():
    from loguru import logger

    f = _featurizer(_narrow_config())
    sharder = ColumnGroupSharder(f._plan)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        sharder.warn_plan_size()
    finally:
        logger.remove(sink_id)
    assert messages == []


def test_warn_plan_size_fires_over_threshold(monkeypatch):
    """Force the threshold under a real config's closure and expect one loud,
    actionable warning naming the offending groups."""
    import featurizer.sharding as sharding_mod
    from loguru import logger

    f = _featurizer(_wide_config())
    sharder = ColumnGroupSharder(f._plan)
    max_closure = max(sharder.plan_size_report().values())
    monkeypatch.setattr(sharding_mod, "PLAN_SIZE_WARN_CLOSURE", max_closure - 1)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        sharder.warn_plan_size()
    finally:
        logger.remove(sink_id)
    assert len(messages) == 1
    assert "Plan-size risk" in messages[0]
    assert "group_" in messages[0]


def test_partition_clusters_same_lineage_columns():
    """Columns sharing a dependency signature occupy adjacent positions, so a
    signature spans at most one group boundary (⇒ its companions are emitted
    by at most 2 groups, not scattered across most of them)."""
    f = _featurizer(_wide_config())
    sharder = ColumnGroupSharder(f._plan, max_columns_per_group=200)

    def signature(col):
        return tuple(
            sorted(
                {
                    src[0]
                    for dep in col.depends_on
                    if (src := f._plan.synth_column_source.get(dep)) is not None
                }
            )
        )

    groups = sharder._partition_columns()
    sig_groups: dict[tuple, set[int]] = {}
    for gidx, cols in enumerate(groups):
        for col in cols:
            sig_groups.setdefault(signature(col), set()).add(gidx)
    for sig, gids in sig_groups.items():
        n_cols = sum(1 for g in groups for c in g if signature(c) == sig)
        max_span = -(-n_cols // sharder.max_columns_per_group) + 1
        assert len(gids) <= max_span, (sig, gids)
        assert sorted(gids) == list(range(min(gids), max(gids) + 1)), sig


def test_partition_is_deterministic_and_lossless():
    f = _featurizer(_wide_config())
    a = ColumnGroupSharder(f._plan)._partition_columns()
    b = ColumnGroupSharder(f._plan)._partition_columns()
    assert [[c.name for c in g] for g in a] == [[c.name for c in g] for g in b]
    flat = [c.name for g in a for c in g]
    assert sorted(flat) == sorted(c.name for c in f._plan.cte_specs[
        f"{f._plan.target.alias}_transform"
    ].columns)
    assert len(flat) == len(set(flat))


def test_window_fn_budget_bounds_each_group():
    """Window-heavy configs close groups early so no group's transform tuple
    carries more window functions than the planning-safe budget (PostgreSQL's
    planning memory is superlinear in same-statement window-function count)."""
    f = _featurizer(
        _wide_config(
            transformations=["identity", "cum_sum", "lag_1", "rolling_mean_7"]
        )
    )
    sharder = ColumnGroupSharder(f._plan, max_window_fns_per_group=50)
    groups = sharder._partition_columns()
    assert len(groups) > 1
    for cols in groups:
        n_windows = sum(ColumnGroupSharder._n_window_fns(c) for c in cols)
        assert n_windows <= 50, n_windows
    flat = [c.name for g in groups for c in g]
    assert len(flat) == len(set(flat))
