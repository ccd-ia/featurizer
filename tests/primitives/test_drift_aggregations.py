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
    """Since ADR-0011-era migration the drift families are set-based: the two
    windows live in the companion pre-pass (``result.preagg.prepass_sql``), not
    in a correlated ``definition``."""
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, col), interval="P1W", relationship=rel)
    assert result is not None and result.preagg is not None
    prepass = result.preagg.prepass_sql
    # recent + baseline windows (each referenced in its FILTER and the WHERE)
    assert prepass.count("daterange") >= 2
    # baseline window is the doubled-back range; both reference the as-of date
    assert "2 * interval 'P1W'" in prepass
    assert "aod.as_of_date" in prepass


def test_kl_drift_is_kl_over_shared_support():
    """Set-based KL: ``count(*) FILTER`` derives recent/baseline shares per
    category in one pass, and the reduction's ``FILTER (rp>0 AND bp>0)``
    reproduces the correlated INNER JOIN's shared support (no self-join)."""
    parent, child, rel = _setup()
    agg = get_aggregations(["kl_drift"])["kl_drift"]
    result = agg(
        parent, child, _feature(child, "category"), interval="P1W", relationship=rel
    )
    assert "LN(rp / NULLIF(bp, 0))" in result.definition
    assert "FILTER (WHERE rp > 0 AND bp > 0)" in result.definition


def test_wasserstein_uses_constant_quantiles():
    """Set-based Wasserstein proxy: per-window quantiles via ordered-set
    ``percentile_cont … FILTER`` (an empty window → NULL, as before)."""
    parent, child, rel = _setup()
    agg = get_aggregations(["wasserstein_drift"])["wasserstein_drift"]
    result = agg(
        parent, child, _feature(child, "amount"), interval="P1W", relationship=rel
    )
    assert "percentile_cont(0.1)" in result.definition
    assert "filter (where is_recent)" in result.definition
    assert "filter (where is_baseline)" in result.definition


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
