"""DB-free shape guards for the set-based pre-aggregation path (ADR-0010).

A stub aggregator opts into ``_build_preagg`` so these guards exercise the
planner's companion-CTE builder without touching any real family (kept out of
the global registry by the fixture, so the execution harness' "every aggregator
accounted for" test is unaffected). They assert the emitted SQL is a single
window pre-pass reduced by a plain ``GROUP BY`` — NOT a correlated subquery —
and that the companion CTE registers with the sharding + materialization
machinery exactly like a plain aggs CTE.
"""

import re
import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.primitives import utils
from featurizer.primitives.aggregations import SubqueryAggregator
from featurizer.primitives.preagg import PreAggSpec, causal_where


class _StubPreAgg(SubqueryAggregator):
    """Numeric aggregator that routes through a companion CTE (avg of a col)."""

    def __init__(self):
        super().__init__(name="stub_preagg")

    def _build_preagg(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        prepass = (
            f"select {ck}, {feature.name} as v from {ct} "
            f"{causal_where(feature, interval)}"
        )
        return PreAggSpec(
            family_key="stubfam",
            interval=interval,
            prepass_sql=prepass,
            reduction="avg(v)",
            reduction_where="v is not null",
        )


@pytest.fixture
def stub_agg():
    """Register the stub only for the duration of a test (never leaks)."""
    utils._AGGREGATIONS["stub_preagg"] = _StubPreAgg()
    try:
        yield
    finally:
        utils._AGGREGATIONS.pop("stub_preagg", None)


def _config(intervals):
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": intervals,
        "aggregations": ["stub_preagg", "count"],
        "transformations": ["identity"],
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


def _featurizer(intervals):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(_config(intervals), handle)
        path = handle.name
    return Featurizer(path, validate=False)


def _cte_block(sql: str, cte_name: str) -> str:
    """The ``<cte_name> as ( … )`` body, balanced to the matching paren."""
    start = sql.index(f"{cte_name} as (")
    depth = 0
    for i in range(sql.index("(", start), len(sql)):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                return sql[start : i + 1]
    raise AssertionError(f"unbalanced parens for {cte_name}")


def test_preagg_emits_companion_cte(stub_agg):
    sql = _featurizer([]).query
    assert "orders_stubfam_all_preaggs_for_customers as (" in sql


def test_preagg_is_not_a_correlated_subquery(stub_agg):
    """The whole point: no per-row ``WHERE sub.<key> = <table>.<key>`` scan."""
    sql = _featurizer([]).query
    assert not re.search(r"sub\.\w+\s*=\s*\w+_transform\.\w+", sql)


def test_preagg_group_by_carries_child_key(stub_agg):
    block = _cte_block(
        _featurizer([]).query, "orders_stubfam_all_preaggs_for_customers"
    )
    assert "group by customer_id" in block
    # the pre-pass is a subquery in FROM, reduced by a plain aggregate
    assert "from (select customer_id" in block
    assert "avg(v)" in block
    assert block.count("(") == block.count(")")


def test_preagg_one_cte_per_interval(stub_agg):
    sql = _featurizer(["P1M"]).query
    assert "orders_stubfam_all_preaggs_for_customers as (" in sql
    assert "orders_stubfam_P1M_preaggs_for_customers as (" in sql
    # exactly one definition of each (name followed by " as (")
    assert sql.count("orders_stubfam_all_preaggs_for_customers as (") == 1
    assert sql.count("orders_stubfam_P1M_preaggs_for_customers as (") == 1


def test_preagg_registers_with_sharding_and_materialization(stub_agg):
    """Companion CTE must ride the ShardableCTE + MaterializationKey paths."""
    plan = _featurizer([])._plan
    cte_name = "orders_stubfam_all_preaggs_for_customers"
    assert cte_name in plan.cte_specs
    assert plan.cte_specs[cte_name].kind == "aggs"
    assert plan.cte_specs[cte_name].key_columns == ["customer_id"]
    assert cte_name in plan.materialization_keys
    assert plan.materialization_keys[cte_name].join_key == "customer_id"
