"""Registry tests ensure primitive discovery stays deterministic."""

import pytest

from featurizer.primitives.abstractions import Entity
from featurizer.primitives.utils import (
    get_aggregations,
    get_transformers,
    list_aggregations,
)


def _make_parent_entity() -> Entity:
    return Entity(alias="patients", table="analytics.patients", id="patient_id")


def _make_child_entity() -> Entity:
    return Entity(
        alias="visits",
        table="analytics.visits",
        id="visit_id",
        temporal_ix="visited_at",
        variables={"duration_minutes": {"type": "numeric"}},
    )


def _get_numeric_feature(entity: Entity):
    return next(ft for ft in entity.features if ft.name == "duration_minutes")


def test_default_aggregations_registered():
    """Registries expose the built-in aggregations for planner wiring."""
    registered = set(list_aggregations())
    assert {"count", "mean", "median", "sum", "stddev"} <= registered


def test_get_aggregations_unknown_raises():
    with pytest.raises(KeyError):
        get_aggregations(["does_not_exist"])


def test_mean_aggregator_returns_new_feature():
    parent = _make_parent_entity()
    child = _make_child_entity()
    feature = _get_numeric_feature(child)
    aggregator = get_aggregations(["mean"])["mean"]

    result = aggregator(parent, child, feature)

    assert result is not None
    assert result is not feature
    assert result.entity is parent
    assert result.name.startswith('"MEAN(')


def test_interval_aggregation_name_annotations():
    parent = _make_parent_entity()
    child = _make_child_entity()
    feature = _get_numeric_feature(child)
    aggregator = get_aggregations(["mean"])["mean"]

    result = aggregator(parent, child, feature, interval="P1D")

    assert result is not None
    assert "interval=P1D" in result.name


def test_transformer_subset_lookup():
    names = [
        "identity",
        "lag_1",
        "rolling_mean_3",
        "rolling_median_7",
        "rolling_iqr_7",
        "ema_7",
        "holt_winters_trend_7",
        "pct_change_1",
    ]
    transformers = get_transformers(names)
    assert set(transformers) == set(names)
