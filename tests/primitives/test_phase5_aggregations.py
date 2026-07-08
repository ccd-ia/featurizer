"""Tests for Phase 5 SubqueryAggregator reductions."""

import pytest

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations, list_aggregations

# name -> the feature kind it operates on
PHASE5 = {
    "theil": "numeric",
    "trimmed_mean_10": "numeric",
    "median_absolute_deviation": "numeric",
    "acf_1": "numeric",
    "variance_ratio": "numeric",
    "cosinor_amplitude_weekly": "numeric",
    "state_volatility": "categorical",
    "transition_matrix_summary": "categorical",
    "rework_count": "categorical",
    "time_in_current_state": "categorical",
}


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
    name = "category" if kind == "categorical" else "amount"
    return next(f for f in child.features if f.name == name)


def _full_sql(result):
    """Definition plus, for a migrated (pre-agg) aggregator, its shared pre-pass
    — where the causal bound / window / percentile SQL lives after ADR-0010."""
    prepass = result.preagg.prepass_sql if result.preagg is not None else ""
    return f"{result.definition} {prepass}"


@pytest.mark.parametrize("name", PHASE5)
def test_registered(name):
    assert name in list_aggregations()


@pytest.mark.parametrize("name,kind", PHASE5.items())
def test_fires_and_is_causally_bounded(name, kind):
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, kind), relationship=rel)
    assert result is not None and result.definition is not None
    assert "<= aod.as_of_date" in _full_sql(result)


@pytest.mark.parametrize("name,kind", PHASE5.items())
def test_interval_variant_uses_daterange(name, kind):
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, kind), interval="P1W", relationship=rel)
    assert result is not None and result.definition is not None
    full = _full_sql(result)
    assert "daterange" in full and "P1W" in full


@pytest.mark.parametrize("name,kind", PHASE5.items())
def test_requires_relationship(name, kind):
    parent, child, _ = _setup()
    agg = get_aggregations([name])[name]
    # SubqueryAggregators return None without a relationship.
    assert agg(parent, child, _feature(child, kind)) is None


@pytest.mark.parametrize(
    "name,fragment",
    [
        ("theil", "LN(val / m)"),
        ("trimmed_mean_10", "between"),
        ("median_absolute_deviation", "abs(v.val - b.med)"),
        ("state_volatility", "IS DISTINCT FROM"),
        ("transition_matrix_summary", "count(DISTINCT (prev, curr))"),
        ("rework_count", "prev = curr"),
        ("time_in_current_state", "max(ts) filter"),
        ("acf_1", "corr(val, lagk)"),
        ("variance_ratio", "var_samp(val) / NULLIF(var_samp(d), 0)"),
        ("cosinor_amplitude_weekly", "regr_slope(val, s)"),
    ],
)
def test_signature_sql_fragment(name, fragment):
    parent, child, rel = _setup()
    kind = PHASE5[name]
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feature(child, kind), relationship=rel)
    assert fragment.lower() in _full_sql(result).lower()


def test_numeric_aggs_reject_categorical():
    parent, child, rel = _setup()
    cat = _feature(child, "categorical")
    for name, kind in PHASE5.items():
        if kind != "numeric":
            continue
        agg = get_aggregations([name])[name]
        assert agg(parent, child, cat, relationship=rel) is None


def test_sequence_aggs_require_temporal_ix():
    parent = Entity(alias="customers", table="c", id="customer_id")
    child = Entity(
        alias="orders",
        table="o",
        id="order_id",
        variables={"category": {"type": "categorical"}},
    )
    rel = Relationship(
        parent=parent, child=child, parent_key="customer_id", child_key="customer_id"
    )
    cat = next(f for f in child.features if f.name == "category")
    for name, kind in PHASE5.items():
        if kind != "categorical":
            continue
        agg = get_aggregations([name])[name]
        assert agg(parent, child, cat, relationship=rel) is None
