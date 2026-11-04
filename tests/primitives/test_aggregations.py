"""DB-free tests for the ``last_value`` aggregation.

``last_value`` reduces a numeric child column to its value on the latest row of
the child's timeline the aggregation sees, skipping NULLs. Two render paths: the
set-based pre-pass (ADR-0010) over ``<child>_transform`` and the correlated
subquery over ``sub``. Their values are checked against a hand-computed oracle
in ``tests/integration/test_last_value.py``; this file pins the SQL shape.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from featurizer import Featurizer
from featurizer import featurizer as featurizer_module
from featurizer.cli import AGGREGATION_CATEGORIES, AGGREGATION_DOCS
from featurizer.primitives import aggregations
from featurizer.primitives.abstractions import Entity, Feature, Relationship
from featurizer.primitives.preagg import PreAggSpec
from featurizer.primitives.utils import get_aggregations, list_aggregations

TIMELINE_DESC = (
    'orders_transform."ordered_at" desc, orders_transform."order_id" desc, '
    'orders_transform."amount" desc, orders_transform."category" desc'
)
SUB_TIMELINE_DESC = (
    'sub."ordered_at" desc, sub."order_id" desc, sub."amount" desc, sub."category" desc'
)


def _entities(*, dated: bool = True) -> tuple[Entity, Entity, Relationship]:
    parent = Entity(alias="customers", table="analytics.customers", id="customer_id")
    child = Entity(
        alias="orders",
        table="analytics.orders",
        id="order_id",
        temporal_ix="ordered_at" if dated else None,
        variables={
            "amount": {"type": "numeric"},
            "category": {"type": "categorical"},
        },
    )
    relationship = Relationship(
        parent=parent, child=child, parent_key="customer_id", child_key="customer_id"
    )
    return parent, child, relationship


def _variable(entity: Entity, name: str) -> Feature:
    return next(feature for feature in entity.features if feature.name == name)


def _last_value(interval: str | None = None, *, dated: bool = True, column="amount"):
    parent, child, relationship = _entities(dated=dated)
    agg = get_aggregations(["last_value"])["last_value"]
    feature = _variable(child, column)
    return agg(parent, child, feature, interval, relationship=relationship), (
        agg,
        child,
        relationship,
        feature,
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_last_value_is_registered_and_categorised() -> None:
    assert "last_value" in list_aggregations()
    assert AGGREGATION_CATEGORIES["last_value"] == "temporal"
    assert AGGREGATION_DOCS["last_value"]["input_types"] == ["numeric"]


def test_last_value_is_not_in_either_default_set() -> None:
    # The config default (what omitting ``aggregations:`` selects) and the
    # module's dict of the historical primitives. Adding it to either would add
    # a column to an existing config and move the render baseline.
    assert "last_value" not in featurizer_module.DEFAULT_AGGREGATIONS
    assert "last_value" not in aggregations.DEFAULT_AGGREGATIONS


# --------------------------------------------------------------------------- #
# Naming and applicability
# --------------------------------------------------------------------------- #


def test_output_name_follows_the_aggregation_pattern() -> None:
    feature, _ = _last_value()
    assert feature is not None
    assert feature.name == '"LAST_VALUE(orders.amount)"'
    assert feature.label == "LAST_VALUE(orders.amount)"
    windowed, _ = _last_value("P30D")
    assert windowed is not None
    assert windowed.name == '"LAST_VALUE(orders.amount|interval=P30D)"'


def test_not_rendered_on_an_undated_child() -> None:
    # No timeline, no "last": the same rule the window transformers follow (#76).
    feature, _ = _last_value(dated=False)
    assert feature is None


def test_not_rendered_on_a_categorical_column() -> None:
    feature, _ = _last_value(column="category")
    assert feature is None


# --------------------------------------------------------------------------- #
# Set-based pre-pass (the path the planner takes)
# --------------------------------------------------------------------------- #


def test_preagg_orders_every_timeline_key_latest_first() -> None:
    feature, _ = _last_value()
    assert feature is not None
    spec = feature.preagg
    assert isinstance(spec, PreAggSpec)
    assert spec.family_key == "last_value:amount"
    assert (
        'row_number() over (partition by orders_transform."customer_id" '
        f"order by {TIMELINE_DESC}) as rn"
    ) in spec.prepass_sql
    assert spec.reduction == "MAX(val)"
    assert feature.definition == spec.reduction
    assert spec.reduction_where == "rn = 1"


def test_preagg_bounds_at_the_as_of_date_and_skips_nulls() -> None:
    feature, _ = _last_value()
    assert feature is not None
    assert feature.preagg is not None
    prepass = feature.preagg.prepass_sql
    assert 'where orders_transform."ordered_at" <= aod.as_of_date' in prepass
    assert prepass.endswith('and orders_transform."amount" is not null')


def test_preagg_under_an_interval_uses_the_shared_window() -> None:
    feature, _ = _last_value("P30D")
    assert feature is not None
    assert feature.preagg is not None
    assert feature.preagg.interval == "P30D"
    prepass = feature.preagg.prepass_sql
    assert (
        "where daterange((aod.as_of_date - interval 'P30D')::date, "
        "aod.as_of_date::date, '[]') @> orders_transform.\"ordered_at\"::date"
    ) in prepass
    assert 'and orders_transform."amount" is not null' in prepass


# --------------------------------------------------------------------------- #
# Correlated subquery (the fallback path)
# --------------------------------------------------------------------------- #


def test_correlated_subquery_takes_the_latest_non_null_row() -> None:
    _, (agg, child, relationship, feature) = _last_value()
    sql = agg._build_subquery_expression(feature, child, relationship, None)
    assert sql == (
        '(select sub."amount" from orders_transform sub '
        'where sub."customer_id" = orders_transform."customer_id" '
        'and sub."ordered_at" <= aod.as_of_date '
        'and sub."amount" is not null '
        f"order by {SUB_TIMELINE_DESC} limit 1)"
    )


def test_correlated_subquery_under_an_interval_uses_the_shared_window() -> None:
    _, (agg, child, relationship, feature) = _last_value()
    sql = agg._build_subquery_expression(feature, child, relationship, "P30D")
    assert (
        "and daterange((aod.as_of_date - interval 'P30D')::date, "
        "aod.as_of_date::date, '[]') @> sub.\"ordered_at\"::date"
    ) in sql
    assert f"order by {SUB_TIMELINE_DESC} limit 1)" in sql


def test_both_paths_order_by_the_same_reversed_timeline() -> None:
    # The pitfall: ``_timeline() + " desc"`` reverses only the last key.
    _, (_, child, relationship, feature) = _last_value()
    ascending = aggregations._timeline("sub", feature, child, relationship)
    descending = aggregations._timeline_desc("sub", feature, child, relationship)
    keys = aggregations._timeline_keys("sub", feature, child, relationship)
    assert ascending == ", ".join(keys)
    assert descending == ", ".join(f"{key} desc" for key in keys)
    assert descending.count(" desc") == len(keys) == 4


# --------------------------------------------------------------------------- #
# Through the planner
# --------------------------------------------------------------------------- #


def _config(aggs: list[str]) -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P30D"],
        "aggregations": aggs,
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "customers",
                "table": "customers",
                "id": "customer_id",
                "variables": {},
            },
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


def _render(tmp_path: Path, aggs: list[str]) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(aggs)))
    return Featurizer(str(path), validate=False).query


def test_planner_routes_last_value_into_one_companion_cte_per_interval(
    tmp_path: Path,
) -> None:
    sql = _render(tmp_path, ["max", "last_value"])
    assert "orders_last_value_amount_all_preaggs_for_customers as (" in sql
    assert "orders_last_value_amount_P30D_preaggs_for_customers as (" in sql
    assert '"LAST_VALUE(orders.amount)"' in sql
    assert '"LAST_VALUE(orders.amount|interval=P30D)"' in sql
    assert "where rn = 1" in sql


def test_a_config_without_last_value_does_not_render_it(tmp_path: Path) -> None:
    sql = _render(tmp_path, ["max"])
    assert "LAST_VALUE" not in sql
    assert "last_value" not in sql


def test_omitting_aggregations_does_not_select_last_value(tmp_path: Path) -> None:
    config = _config(["max"])
    del config["aggregations"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert "last_value" not in Featurizer(str(path), validate=False).aggregations
