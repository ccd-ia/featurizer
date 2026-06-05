"""Tests for as-of state aggregations (Phase 3): recency, tenure, hazard."""

import pytest

from featurizer.primitives.abstractions import Entity
from featurizer.primitives.utils import get_aggregations, list_aggregations

ASOF_NAMES = ["recency", "tenure", "age_in_system", "inter_event_hazard_proxy"]


def _parent():
    return Entity(alias="customers", table="analytics.customers", id="customer_id")


def _child():
    return Entity(
        alias="orders",
        table="analytics.orders",
        id="order_id",
        temporal_ix="ordered_at",
        variables={"amount": {"type": "numeric"}},
    )


def _amount(entity):
    return next(f for f in entity.features if f.name == "amount")


@pytest.mark.parametrize("name", ASOF_NAMES)
def test_registered(name):
    assert name in list_aggregations()


@pytest.mark.parametrize(
    "name,sql",
    [
        ("recency", "max(ordered_at)"),
        ("tenure", "min(ordered_at)"),
        ("age_in_system", "min(ordered_at)"),
        ("inter_event_hazard_proxy", "count(*)"),
    ],
)
def test_definition_sql(name, sql):
    parent, child = _parent(), _child()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, child.temporal_ix)
    assert result is not None
    assert result.definition is not None
    assert sql in result.definition
    # Causal safety: the backward bound references the as-of date.
    assert "aod.as_of_date" in result.definition


@pytest.mark.parametrize("name", ASOF_NAMES)
def test_interval_variant_bounds_window(name):
    parent, child = _parent(), _child()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, child.temporal_ix, interval="P1W")
    assert result is not None
    assert result.definition is not None
    assert "daterange" in result.definition
    assert "P1W" in result.definition


@pytest.mark.parametrize("name", ASOF_NAMES)
def test_only_fires_on_temporal_ix(name):
    """A non-temporal feature (e.g. a numeric variable) yields None."""
    parent, child = _parent(), _child()
    agg = get_aggregations([name])[name]
    assert agg(parent, child, _amount(child)) is None


@pytest.mark.parametrize("name", ASOF_NAMES)
def test_no_temporal_ix_returns_none(name):
    parent = _parent()
    child = Entity(
        alias="orders",
        table="o",
        id="order_id",
        variables={"amount": {"type": "numeric"}},
    )
    agg = get_aggregations([name])[name]
    assert agg(parent, child, _amount(child)) is None


def test_recency_name_format():
    parent, child = _parent(), _child()
    agg = get_aggregations(["recency"])["recency"]
    result = agg(parent, child, child.temporal_ix)
    assert "RECENCY(orders.ordered_at)" in result.name
