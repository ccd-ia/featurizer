"""Tests for config-driven primitive selection (Phase 3.5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from featurizer import Featurizer, validate_config
from featurizer.featurizer import DEFAULT_AGGREGATIONS, DEFAULT_TRANSFORMATIONS


def _write(config_text: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_text)
        f.flush()
        return f.name


_BASE = """
target: users
max_depth: 1
intervals: []
entities:
  - alias: users
    table: users
    id: user_id
    temporal_ix: created_at
    variables:
      age: {type: numeric}
"""


def test_aggregations_selection_overrides_defaults():
    path = _write(_BASE + "aggregations: [sum]\n")
    f = Featurizer(path)
    Path(path).unlink()
    assert set(f.aggregations.keys()) == {"sum"}


def test_transformations_selection_overrides_defaults():
    path = _write(_BASE + "transformations: [identity]\n")
    f = Featurizer(path)
    Path(path).unlink()
    assert set(f.transformations.keys()) == {"identity"}


def test_missing_keys_fall_back_to_defaults():
    path = _write(_BASE)
    f = Featurizer(path)
    Path(path).unlink()
    assert set(f.aggregations.keys()) == set(DEFAULT_AGGREGATIONS)
    # the expanded default set includes the new as-of-state primitives
    assert {"recency", "tenure"} <= set(f.aggregations.keys())


def test_unknown_aggregation_raises_with_suggestion():
    path = _write(_BASE + "aggregations: [recencey]\n")
    with pytest.raises(ValueError, match="Unknown aggregation primitive"):
        Featurizer(path)
    Path(path).unlink()


def test_unknown_transformation_raises():
    path = _write(_BASE + "transformations: [identty]\n")
    with pytest.raises(ValueError, match="Unknown transformation primitive"):
        Featurizer(path)
    Path(path).unlink()


def test_non_list_aggregations_raises():
    path = _write(_BASE + "aggregations: sum\n")
    with pytest.raises(ValueError, match="'aggregations' must be a list"):
        Featurizer(path)
    Path(path).unlink()


def test_validate_config_suggests_correction():
    path = _write(_BASE + "aggregations: [recencey]\n")
    result = validate_config(path)
    Path(path).unlink()
    assert not result.is_valid
    assert any("recency" in (e.suggestion or "") for e in result.errors)


# Empty-list semantics: an explicit `[]` suppresses that feature layer; it is
# NOT the same as omitting the key (which applies the module defaults).
# `transformations: []` passes features through unchanged — the contract is
# byte-identical output to `transformations: [identity]`, the workaround
# downstream shipped before this was fixed.

_PARENT_CHILD = """
target: users
max_depth: 2
intervals: [P1M]
entities:
  - alias: users
    table: users
    id: user_id
    temporal_ix: created_at
    variables:
      age: {type: numeric}
  - alias: orders
    table: orders
    id: order_id
    temporal_ix: ordered_at
    variables:
      amount: {type: numeric}
relationships:
  - parent: {entity: users, key: user_id}
    child: {entity: orders, key: user_id}
"""

_TRANSFORM_MARKERS = ("CUM_SUM(", "ABS(", "LAG_", "ROLLING_", "EMA_", "PCT_CHANGE_")


def _query_for(config_text: str) -> str:
    path = _write(config_text)
    query = Featurizer(path).query
    Path(path).unlink()
    return query


def test_empty_transformations_suppresses_transform_layer():
    query = _query_for(_PARENT_CHILD + "transformations: []\n")
    for marker in _TRANSFORM_MARKERS:
        assert marker not in query
    # The aggregation layer is untouched: default aggs still present.
    assert "MEAN(" in query


def test_empty_transformations_identical_to_identity_workaround():
    empty = _query_for(_PARENT_CHILD + "transformations: []\n")
    identity = _query_for(_PARENT_CHILD + "transformations: [identity]\n")
    assert empty == identity


def test_null_transformations_applies_defaults():
    # `transformations:` with no value parses to None — same as an absent key.
    # (Registry-level assertion: the full default matrix at depth 2 is too wide
    # to render as one query, and rendering is not what's under test here.)
    path = _write(_PARENT_CHILD + "transformations: null\n")
    f = Featurizer(path)
    Path(path).unlink()
    assert set(f.transformations.keys()) == set(DEFAULT_TRANSFORMATIONS)


def test_empty_aggregations_suppresses_aggregation_layer():
    # Legal-but-weird by decision: zero aggregation features, no agg CTEs.
    # Direct variables and the transform layer are untouched.
    query = _query_for(_PARENT_CHILD + "aggregations: []\n")
    assert "MEAN(" not in query
    assert "_aggs_for_" not in query
    assert "CUM_SUM(" in query


def test_null_aggregations_applies_defaults():
    path = _write(_PARENT_CHILD + "aggregations: null\n")
    f = Featurizer(path)
    Path(path).unlink()
    assert set(f.aggregations.keys()) == set(DEFAULT_AGGREGATIONS)


def test_both_layers_empty_leaves_direct_passthrough_only():
    query = _query_for(_PARENT_CHILD + "aggregations: []\ntransformations: []\n")
    assert "_aggs_for_" not in query
    for marker in _TRANSFORM_MARKERS:
        assert marker not in query
    # The direct variable passes through the transform CTE unchanged.
    assert "age as age" in query
