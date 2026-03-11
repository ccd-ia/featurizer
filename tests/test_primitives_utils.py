"""Tests for primitive registration and discovery."""

import pytest

from featurizer.primitives.utils import (
    get_aggregations,
    get_transformers,
    list_aggregations,
    list_transformations,
    register_aggregation,
    register_transformer,
)


def test_register_aggregation_with_empty_name_raises():
    """ValueError when registering aggregation with empty name."""
    with pytest.raises(ValueError, match="Aggregation name must be a non-empty string"):
        register_aggregation("", lambda: None)


def test_register_aggregation_with_non_callable_raises():
    """TypeError when registering non-callable aggregation."""
    with pytest.raises(TypeError, match="must be callable"):
        register_aggregation("test", "not callable")


def test_register_transformer_with_empty_name_raises():
    """ValueError when registering transformer with empty name."""
    with pytest.raises(ValueError, match="Transformer name must be a non-empty string"):
        register_transformer("", lambda: None)


def test_register_transformer_with_non_callable_raises():
    """TypeError when registering non-callable transformer."""
    with pytest.raises(TypeError, match="must be callable"):
        register_transformer("test", "not callable")


def test_get_aggregations_all():
    """get_aggregations(None) returns all registered."""
    all_aggs = get_aggregations(None)
    assert isinstance(all_aggs, dict)
    assert len(all_aggs) > 0


def test_get_transformers_all():
    """get_transformers(None) returns all registered."""
    all_trans = get_transformers(None)
    assert isinstance(all_trans, dict)
    assert len(all_trans) > 0


def test_get_aggregations_unknown_raises():
    """KeyError when requesting unknown aggregation."""
    with pytest.raises(KeyError, match="Unknown aggregations requested"):
        get_aggregations(["nonexistent_aggregation"])


def test_get_transformers_unknown_raises():
    """KeyError when requesting unknown transformer."""
    with pytest.raises(KeyError, match="Unknown transformers requested"):
        get_transformers(["nonexistent_transformer"])


def test_list_aggregations_returns_sorted():
    """list_aggregations returns sorted list of names."""
    names = list(list_aggregations())
    assert isinstance(names, list)
    assert names == sorted(names)
    assert "count" in names
    assert "mean" in names


def test_list_transformations_returns_sorted():
    """list_transformations returns sorted list of names."""
    names = list(list_transformations())
    assert isinstance(names, list)
    assert names == sorted(names)
    assert "identity" in names
