"""Every correlated-subquery fallback renders, and renders causally (issue #83).

ADR-0010 moved most ``SubqueryAggregator`` families onto a set-based pre-pass,
and the planner always takes it when ``_build_preagg`` returns a spec. The
correlated ``_build_subquery_expression`` each of them keeps is then dead code
in every rendered query, so nothing renders it, and nothing noticed when
``sequence_entropy``'s called ``_timeline`` with one argument too many: the
fallback raised ``TypeError`` the first time a test turned the pre-pass off.

This sweep calls the fallback directly, for every registered aggregation that
is a ``SubqueryAggregator`` (from the registry, not a hand list), on a dated
child, over the whole history and over an interval. It must return SQL, and
that SQL must carry the backward causal bound: ``<= aod.as_of_date``, or the
interval's ``daterange`` window, whose upper bound is the as-of date.

The values the fallback computes are compared with the pre-pass's in
``tests/integration/test_correlated_fallback_parity.py``.
"""

from __future__ import annotations

import pytest

from featurizer.primitives.abstractions import Entity, Feature, Relationship
from featurizer.primitives.aggregations import SubqueryAggregator
from featurizer.primitives.utils import get_aggregations, list_aggregations

INTERVALS = (None, "P30D")

# Not rendered here, by name, each with the configuration it needs and this
# child does not declare. None of them has a set-based path, so the correlated
# subquery IS the path the planner takes, and the registry-wide executing
# sweeps (tests/integration/test_all_aggregators_execution.py and the sweep
# matrix in tests/test_sweep_matrix_coverage.py) already run it.
SKIPPED = {
    "bbox_area": "spatial: reads the child's spatial_ix (lat / lon)",
    "distance_travelled": "spatial: reads the child's spatial_ix (lat / lon)",
    "radius_of_gyration": "spatial: reads the child's spatial_ix (lat / lon)",
    "spatial_std": "spatial: reads the child's spatial_ix (lat / lon)",
    "cross_type_latency": "predicates: needs predicates {a, b} on the variable",
    "first_passage_time": "predicates: needs a predicate {target} on the variable",
    "right_censoring_indicator": (
        "predicates: needs a predicate {terminal} on the variable"
    ),
}


def _entities() -> tuple[Entity, Entity, Relationship]:
    """The parent / dated child of ``tests/primitives/test_aggregations.py``."""
    parent = Entity(alias="customers", table="analytics.customers", id="customer_id")
    child = Entity(
        alias="orders",
        table="analytics.orders",
        id="order_id",
        temporal_ix="ordered_at",
        variables={
            "amount": {"type": "numeric"},
            "category": {"type": "categorical"},
        },
    )
    relationship = Relationship(
        parent=parent, child=child, parent_key="customer_id", child_key="customer_id"
    )
    return parent, child, relationship


def _input(child: Entity, input_type: str) -> Feature | None:
    """The child column an aggregator of ``input_type`` reduces.

    An ``index`` aggregation (the gap family, ``burstiness``) reads the
    temporal index, which is the feature the planner hands it.
    """
    if input_type == "index":
        return child.temporal_ix
    return next((f for f in child.features if f.type == input_type), None)


def _subquery_aggregators() -> dict[str, SubqueryAggregator]:
    aggregations = get_aggregations(list_aggregations())
    return {
        name: agg
        for name, agg in sorted(aggregations.items())
        if isinstance(agg, SubqueryAggregator)
    }


def _cases():
    for name, agg in _subquery_aggregators().items():
        for input_type in agg.input_types:
            for interval in INTERVALS:
                marks = (
                    [pytest.mark.skip(reason=f"{name}: {SKIPPED[name]}")]
                    if name in SKIPPED
                    else []
                )
                yield pytest.param(
                    name,
                    input_type,
                    interval,
                    id=f"{name}-{input_type}-{interval or 'all'}",
                    marks=marks,
                )


@pytest.mark.parametrize(("name", "input_type", "interval"), list(_cases()))
def test_the_correlated_fallback_renders_with_its_causal_bound(
    name: str, input_type: str, interval: str | None
) -> None:
    agg = _subquery_aggregators()[name]
    _, child, relationship = _entities()
    feature = _input(child, input_type)
    assert feature is not None, f"the child declares no {input_type} column"

    sql = agg._build_subquery_expression(feature, child, relationship, interval)

    assert isinstance(sql, str) and sql.strip(), f"{name} rendered no SQL"
    assert "<= aod.as_of_date" in sql or "daterange" in sql, (
        f"{name}'s correlated subquery reads the child with no causal bound"
    )
    if interval is not None:
        assert "daterange" in sql, f"{name} ignores the {interval} window"


def test_the_skip_list_names_only_registered_subquery_aggregators() -> None:
    """A stale entry would hide nothing and say something false."""
    assert set(SKIPPED) <= set(_subquery_aggregators())


def test_the_sweep_covers_every_registered_subquery_aggregator() -> None:
    covered = {case.values[0] for case in _cases()}
    assert covered == set(_subquery_aggregators())
    assert covered, "the registry exposes no SubqueryAggregator"
