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


def test_oversized_child_warn_oversized_emits_named_bound():
    """``warn_oversized`` logs a bound naming each offending CTE without raising.

    loguru holds the import-time ``sys.stderr`` reference, so ``capsys`` cannot
    see its output; add a temporary loguru sink to capture the warnings instead.
    """
    from loguru import logger

    f = _featurizer(_oversized_child_config())
    sharder = ColumnGroupSharder(f._plan)

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        sharder.warn_oversized()
    finally:
        logger.remove(sink_id)

    joined = "".join(messages)
    assert "items_aggs_for_orders" in joined
    assert "1664" in joined
