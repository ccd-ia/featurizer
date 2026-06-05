"""Tests for Phase 4 transformers: diff2, diff3, cumprod."""

import pytest

from featurizer.primitives.abstractions import Entity
from featurizer.primitives.utils import get_transformers, list_transformations

PHASE4_NAMES = ["diff2", "diff3", "cumprod"]


def _entity():
    return Entity(
        alias="sensors",
        table="analytics.sensors",
        id="sensor_id",
        temporal_ix="ts",
        variables={"value": {"type": "numeric"}, "label": {"type": "categorical"}},
    )


def _entity_no_temporal():
    return Entity(
        alias="sensors",
        table="analytics.sensors",
        id="sensor_id",
        variables={"value": {"type": "numeric"}},
    )


def _feature(entity, name):
    return next(f for f in entity.features if f.name == name)


@pytest.mark.parametrize("name", PHASE4_NAMES)
def test_registered(name):
    assert name in list_transformations()


def test_diff2_definition():
    e = _entity()
    tx = get_transformers(["diff2"])["diff2"]
    result = tx(e, _feature(e, "value"))
    assert result.definition is not None
    assert "lag(value, 1)" in result.definition
    assert "lag(value, 2)" in result.definition
    assert "- 2*(" in result.definition
    assert "partition by sensor_id order by ts" in result.definition


def test_diff3_definition():
    e = _entity()
    tx = get_transformers(["diff3"])["diff3"]
    result = tx(e, _feature(e, "value"))
    assert result.definition is not None
    for k in ("lag(value, 1)", "lag(value, 2)", "lag(value, 3)"):
        assert k in result.definition
    assert "3*(" in result.definition


def test_cumprod_definition_is_log_sum_exp_and_guarded():
    e = _entity()
    tx = get_transformers(["cumprod"])["cumprod"]
    result = tx(e, _feature(e, "value"))
    assert result.definition is not None
    assert "exp(sum(ln(value))" in result.definition
    assert "case when min(value)" in result.definition  # positivity guard


@pytest.mark.parametrize("name", PHASE4_NAMES)
def test_non_numeric_returns_feature_unchanged(name):
    e = _entity()
    tx = get_transformers([name])[name]
    label = _feature(e, "label")  # categorical
    assert tx(e, label) is label


@pytest.mark.parametrize("name", PHASE4_NAMES)
def test_no_temporal_ix_returns_none(name):
    e = _entity_no_temporal()
    tx = get_transformers([name])[name]
    assert tx(e, _feature(e, "value")) is None


def test_transformers_return_new_feature_instances():
    """Transformers must not mutate the input feature (hashing invariant)."""
    e = _entity()
    feat = _feature(e, "value")
    for name in PHASE4_NAMES:
        tx = get_transformers([name])[name]
        result = tx(e, feat)
        assert result is not feat
        assert result.name != feat.name
