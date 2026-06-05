"""Tests for config-driven primitive selection (Phase 3.5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from featurizer import Featurizer, validate_config
from featurizer.featurizer import DEFAULT_AGGREGATIONS


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
