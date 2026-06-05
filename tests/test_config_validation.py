"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest

from featurizer import Featurizer


def test_missing_config_file_raises():
    """FileNotFoundError when config file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Featurizer("/nonexistent/path.yaml")


def test_invalid_yaml_raises():
    """ValueError when YAML is malformed."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [\n")
        f.flush()

        with pytest.raises(ValueError, match="Invalid YAML"):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_missing_target_key_raises():
    """ValueError when 'target' key is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
max_depth: 2
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|Missing required keys.*target"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_missing_max_depth_key_raises():
    """ValueError when 'max_depth' key is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|Missing required keys.*max_depth"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_missing_intervals_key_raises():
    """ValueError when 'intervals' key is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 2
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|Missing required keys.*intervals"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_missing_entities_key_raises():
    """ValueError when 'entities' key is missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 2
intervals: [P1D]
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|Missing required keys.*entities"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_empty_target_string_raises():
    """ValueError when target is empty string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: ""
max_depth: 2
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(ValueError, match="'target' must be a non-empty string"):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_non_string_target_raises():
    """ValueError when target is not a string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: 123
max_depth: 2
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(ValueError, match="'target' must be a non-empty string"):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_non_integer_max_depth_raises():
    """ValueError when max_depth is not an integer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: "two"
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|'max_depth' must be.*integer"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_zero_max_depth_raises():
    """ValueError when max_depth is zero."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 0
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|'max_depth' must be.*positive"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_negative_max_depth_raises():
    """ValueError when max_depth is negative."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: -1
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|'max_depth' must be.*positive"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_empty_entities_list_raises():
    """ValueError when entities list is empty."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 2
intervals: [P1D]
entities: []
""")
        f.flush()

        with pytest.raises(
            ValueError,
            match="validation failed|entities.*cannot be empty|at least one entity",
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_non_list_intervals_raises():
    """ValueError when intervals is not a list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 2
intervals: "P1D"
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(ValueError, match="'intervals' must be a list"):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_non_list_relationships_raises():
    """ValueError when relationships is not a list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 2
intervals: [P1D]
entities:
  - alias: test
    table: test.table
    id: test_id
relationships: "not a list"
""")
        f.flush()

        with pytest.raises(ValueError, match="'relationships' must be a list"):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_missing_relationships_defaults_to_empty_list():
    """Missing relationships defaults to empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 2
intervals: []
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        featurizer = Featurizer(f.name)
        assert featurizer.graph.relationships == []

        Path(f.name).unlink()


def test_unknown_target_entity_raises():
    """ValueError when target entity doesn't exist in entities."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: nonexistent
max_depth: 2
intervals: []
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        with pytest.raises(
            ValueError, match="validation failed|Target entity.*not found"
        ):
            Featurizer(f.name)

        Path(f.name).unlink()


def test_env_debug_flag_enabled():
    """Debug mode can be enabled via environment variable."""
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 1
intervals: []
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        # Test various truthy values
        for value in ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]:
            os.environ["FEATURIZER_DEBUG"] = value
            featurizer = Featurizer(f.name)
            assert featurizer._debug_enabled is True
            del os.environ["FEATURIZER_DEBUG"]

        # Test falsy values
        for value in ["0", "false", "no", "off", ""]:
            os.environ["FEATURIZER_DEBUG"] = value
            featurizer = Featurizer(f.name)
            assert featurizer._debug_enabled is False
            if "FEATURIZER_DEBUG" in os.environ:
                del os.environ["FEATURIZER_DEBUG"]

        Path(f.name).unlink()


def test_debug_parameter_overrides_env():
    """Debug parameter takes precedence."""
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 1
intervals: []
entities:
  - alias: test
    table: test.table
    id: test_id
""")
        f.flush()

        os.environ["FEATURIZER_DEBUG"] = "0"
        featurizer = Featurizer(f.name, debug=True)
        assert featurizer._debug_enabled is True
        del os.environ["FEATURIZER_DEBUG"]

        Path(f.name).unlink()


def test_to_dataframe_without_target_id_raises():
    """ValueError when target entity has no ID and to_dataframe is called."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
target: test
max_depth: 1
intervals: []
entities:
  - alias: test
    table: test.table
    id: ~
""")
        f.flush()

        featurizer = Featurizer(f.name)

        with pytest.raises(
            ValueError, match="Target entity 'test' does not define a primary id"
        ):
            featurizer.to_dataframe()

        Path(f.name).unlink()
