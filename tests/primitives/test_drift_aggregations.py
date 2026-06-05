"""Tests for two-window distributional drift aggregators (Phase 7)."""

import pytest

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations, list_aggregations

DRIFT = {"kl_drift": "category", "wasserstein_drift": "amount"}


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


def _feature(child, col):
    return next(f for f in child.features if f.name == col)


@pytest.mark.parametrize("name", DRIFT)
def test_registered(name):
    assert name in list_aggregations()


@pytest.mark.parametrize("name,col", DRIFT.items())
def test_interval_only(name, col):
    """Drift is interval-only: it returns None on the non-interval pass."""
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    assert agg(parent, child, _feature(child, col), relationship=rel) is None


@pytest.mark.parametrize("name,col", DRIFT.items())
def test_two_windows_both_backward(name, col):
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, col), interval="P1W", relationship=rel)
    assert result is not None and result.definition is not None
    # recent + baseline windows
    assert result.definition.count("daterange") == 2
    # baseline window is the doubled-back range; both reference the as-of date
    assert "2 * interval 'P1W'" in result.definition
    assert "aod.as_of_date" in result.definition


def test_kl_drift_is_kl_over_shared_support():
    parent, child, rel = _setup()
    agg = get_aggregations(["kl_drift"])["kl_drift"]
    result = agg(
        parent, child, _feature(child, "category"), interval="P1W", relationship=rel
    )
    assert "LN(r.p / NULLIF(b.p, 0))" in result.definition
    assert "JOIN" in result.definition.upper()


def test_wasserstein_uses_constant_quantiles():
    parent, child, rel = _setup()
    agg = get_aggregations(["wasserstein_drift"])["wasserstein_drift"]
    result = agg(
        parent, child, _feature(child, "amount"), interval="P1W", relationship=rel
    )
    assert "percentile_cont(0.1)" in result.definition
    assert "ABS(r.q10 - b.q10)" in result.definition


def test_type_gating():
    parent, child, rel = _setup()
    kl = get_aggregations(["kl_drift"])["kl_drift"]
    ws = get_aggregations(["wasserstein_drift"])["wasserstein_drift"]
    # kl_drift is categorical-only; wasserstein is numeric-only
    assert (
        kl(parent, child, _feature(child, "amount"), interval="P1W", relationship=rel)
        is None
    )
    assert (
        ws(parent, child, _feature(child, "category"), interval="P1W", relationship=rel)
        is None
    )
