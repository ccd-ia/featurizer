"""Causal-safety regression tests for SubqueryAggregators.

A correlated subquery reads `<child>_transform` directly, so it must carry its
own backward bound (`<= aod.as_of_date`, or the daterange window) — the outer
aggregation WHERE does not reach into the subquery. These tests lock that in for
every subquery aggregator (the prior leak was the missing non-interval bound).
"""

import pytest

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations

SUBQUERY_AGGS = [
    ("gap_mean", "temporal"),
    ("gap_stddev", "temporal"),
    ("gap_min", "temporal"),
    ("gap_max", "temporal"),
    ("gap_cv", "temporal"),
    ("burstiness", "temporal"),
    ("entropy", "categorical"),
    ("hhi", "categorical"),
    ("ngram_2_freq", "categorical"),
    ("ngram_3_freq", "categorical"),
    ("sequence_entropy", "categorical"),
    ("longest_streak", "categorical"),
    ("gini", "numeric"),
]


def _setup():
    parent = Entity(alias="customers", table="c", id="customer_id")
    child = Entity(
        alias="orders",
        table="o",
        id="order_id",
        temporal_ix="ordered_at",
        variables={"category": {"type": "categorical"}, "amount": {"type": "numeric"}},
    )
    rel = Relationship(
        parent=parent, child=child, parent_key="customer_id", child_key="customer_id"
    )
    return parent, child, rel


def _feature(child, kind):
    if kind == "temporal":
        return child.temporal_ix
    name = "category" if kind == "categorical" else "amount"
    return next(f for f in child.features if f.name == name)


@pytest.mark.parametrize("name,kind", SUBQUERY_AGGS)
def test_subquery_bounded_without_interval(name, kind):
    """Non-interval subquery must bound rows at `<= aod.as_of_date`."""
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, kind), relationship=rel)
    assert result is not None and result.definition is not None
    assert "<= aod.as_of_date" in result.definition


@pytest.mark.parametrize("name,kind", SUBQUERY_AGGS)
def test_subquery_bounded_with_interval(name, kind):
    """Interval subquery must bound rows with the as-of-anchored daterange."""
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, kind), interval="P1W", relationship=rel)
    assert result is not None and result.definition is not None
    assert "daterange" in result.definition
    assert "P1W" in result.definition
