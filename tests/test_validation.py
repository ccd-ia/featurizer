"""Tests for enhanced configuration validation."""

import tempfile
from pathlib import Path

import pytest

from featurizer import validate_config, ValidationError, ValidationResult, ValidationWarning
from featurizer.validation import ConfigValidator


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_valid_config_passes(self):
        """Valid configuration returns no errors."""
        config = {
            "target": "users",
            "max_depth": 2,
            "intervals": ["P7D", "P1M"],
            "entities": [
                {"alias": "users", "table": "users", "id": "user_id"},
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_required_keys(self):
        """Missing required keys generates errors."""
        config = {"target": "users"}  # Missing max_depth, intervals, entities

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("Missing required keys" in error.message for error in result.errors)

    def test_invalid_target_type(self):
        """Non-string target generates error."""
        config = {
            "target": 123,
            "max_depth": 2,
            "intervals": [],
            "entities": [{"alias": "test", "table": "test", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("target" in error.message.lower() for error in result.errors)

    def test_empty_target_string(self):
        """Empty target string generates error."""
        config = {
            "target": "",
            "max_depth": 2,
            "intervals": [],
            "entities": [{"alias": "test", "table": "test", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid

    def test_negative_max_depth(self):
        """Negative max_depth generates error."""
        config = {
            "target": "test",
            "max_depth": -1,
            "intervals": [],
            "entities": [{"alias": "test", "table": "test", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("positive" in error.message.lower() for error in result.errors)

    def test_zero_max_depth(self):
        """Zero max_depth generates error."""
        config = {
            "target": "test",
            "max_depth": 0,
            "intervals": [],
            "entities": [{"alias": "test", "table": "test", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid


class TestIntervalValidation:
    """Tests for ISO 8601 interval validation."""

    def test_valid_intervals_pass(self):
        """Valid ISO 8601 intervals pass validation."""
        valid_intervals = ["P1D", "P7D", "P1W", "P1M", "P1Y", "P2W", "PT1H", "PT30M"]

        for interval in valid_intervals:
            config = {
                "target": "test",
                "max_depth": 1,
                "intervals": [interval],
                "entities": [{"alias": "test", "table": "test", "id": "id"}],
            }

            validator = ConfigValidator()
            result = validator.validate(config)

            assert result.is_valid, f"Interval '{interval}' should be valid"

    def test_invalid_intervals_generate_errors(self):
        """Invalid interval formats generate errors."""
        invalid_intervals = [
            "1 week",
            "7 days",
            "1week",
            "P",
            "",
            "PXD",
            "P-1D",
        ]

        for interval in invalid_intervals:
            config = {
                "target": "test",
                "max_depth": 1,
                "intervals": [interval],
                "entities": [{"alias": "test", "table": "test", "id": "id"}],
            }

            validator = ConfigValidator()
            result = validator.validate(config)

            assert not result.is_valid, f"Interval '{interval}' should be invalid"
            assert any("ISO 8601" in error.message for error in result.errors)


class TestVariableTypeValidation:
    """Tests for variable type validation."""

    def test_valid_variable_types_pass(self):
        """Valid variable types pass validation."""
        valid_types = [
            "numeric",
            "categorical",
            "text",
            "boolean",
            "date",
            "timestamp",
            "index",
        ]

        for var_type in valid_types:
            config = {
                "target": "test",
                "max_depth": 1,
                "intervals": [],
                "entities": [
                    {
                        "alias": "test",
                        "table": "test",
                        "id": "id",
                        "variables": {"col1": {"type": var_type}},
                    }
                ],
            }

            validator = ConfigValidator()
            result = validator.validate(config)

            assert result.is_valid, f"Variable type '{var_type}' should be valid"

    def test_invalid_variable_type_generates_error(self):
        """Invalid variable type generates error."""
        config = {
            "target": "test",
            "max_depth": 1,
            "intervals": [],
            "entities": [
                {
                    "alias": "test",
                    "table": "test",
                    "id": "id",
                    "variables": {"col1": {"type": "invalid_type"}},
                }
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("Invalid variable type" in error.message for error in result.errors)

    def test_typo_in_variable_type_suggests_correction(self):
        """Typo in variable type suggests similar valid type."""
        config = {
            "target": "test",
            "max_depth": 1,
            "intervals": [],
            "entities": [
                {
                    "alias": "test",
                    "table": "test",
                    "id": "id",
                    "variables": {"col1": {"type": "numric"}},  # Typo
                }
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        error_messages = result.format_errors()
        assert "numeric" in error_messages.lower()  # Should suggest "numeric"


class TestSemanticValidation:
    """Tests for semantic cross-field validation."""

    def test_unknown_target_entity_generates_error(self):
        """Target referencing unknown entity generates error."""
        config = {
            "target": "unknown",
            "max_depth": 1,
            "intervals": [],
            "entities": [{"alias": "known", "table": "known", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("Target entity" in error.message for error in result.errors)
        assert any("not found" in error.message for error in result.errors)

    def test_typo_in_target_suggests_correction(self):
        """Typo in target entity suggests similar entity."""
        config = {
            "target": "userz",  # Typo
            "max_depth": 1,
            "intervals": [],
            "entities": [{"alias": "users", "table": "users", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        error_messages = result.format_errors()
        assert "users" in error_messages  # Should suggest "users"

    def test_unknown_relationship_entity_generates_error(self):
        """Relationship referencing unknown entity generates error."""
        config = {
            "target": "users",
            "max_depth": 2,
            "intervals": [],
            "entities": [
                {"alias": "users", "table": "users", "id": "user_id"},
                {"alias": "orders", "table": "orders", "id": "order_id"},
            ],
            "relationships": [
                {
                    "parent": {"entity": "users", "key": "user_id"},
                    "child": {"entity": "unknown", "key": "user_id"},  # Unknown
                }
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("unknown entity" in error.message.lower() for error in result.errors)

    def test_duplicate_entity_aliases_generate_error(self):
        """Duplicate entity aliases generate error."""
        config = {
            "target": "users",
            "max_depth": 1,
            "intervals": [],
            "entities": [
                {"alias": "users", "table": "users1", "id": "id1"},
                {"alias": "users", "table": "users2", "id": "id2"},  # Duplicate
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert not result.is_valid
        assert any("Duplicate" in error.message for error in result.errors)

    def test_temporal_join_without_temporal_ix_warns(self):
        """Temporal join without temporal_ix generates warning."""
        config = {
            "target": "patients",
            "max_depth": 2,
            "intervals": [],
            "entities": [
                {
                    "alias": "patients",
                    "table": "patients",
                    "id": "patient_id",
                    # Missing temporal_ix
                },
                {
                    "alias": "care_plans",
                    "table": "care_plans",
                    "id": "plan_id",
                    "temporal_ix": "effective_at",
                },
            ],
            "relationships": [
                {
                    "parent": {"entity": "care_plans", "key": "patient_id"},
                    "child": {"entity": "patients", "key": "patient_id"},
                    "temporal": {"mode": "as_of"},
                }
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        # Should be valid (warnings, not errors)
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("temporal" in warning.message.lower() for warning in result.warnings)


class TestCircularRelationshipDetection:
    """Tests for circular relationship detection."""

    def test_circular_relationship_generates_warning(self):
        """Circular relationships generate warnings."""
        config = {
            "target": "A",
            "max_depth": 3,
            "intervals": [],
            "entities": [
                {"alias": "A", "table": "a", "id": "id"},
                {"alias": "B", "table": "b", "id": "id"},
            ],
            "relationships": [
                {"parent": {"entity": "A", "key": "id"}, "child": {"entity": "B", "key": "a_id"}},
                {"parent": {"entity": "B", "key": "id"}, "child": {"entity": "A", "key": "b_id"}},
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        # Should be valid but with warning
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("circular" in warning.message.lower() for warning in result.warnings)


class TestBestPracticesValidation:
    """Tests for best practices warnings."""

    def test_very_high_max_depth_warns(self):
        """Very high max_depth generates warning."""
        config = {
            "target": "test",
            "max_depth": 10,  # Very high
            "intervals": [],
            "entities": [{"alias": "test", "table": "test", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) > 0
        assert any("max_depth" in warning.message.lower() for warning in result.warnings)

    def test_many_intervals_warns(self):
        """Many intervals generate warning."""
        config = {
            "target": "test",
            "max_depth": 2,
            "intervals": [f"P{i}D" for i in range(1, 20)],  # 19 intervals
            "entities": [{"alias": "test", "table": "test", "id": "id"}],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid  # Valid but with warning
        assert len(result.warnings) > 0
        assert any("interval" in warning.message.lower() for warning in result.warnings)


class TestValidationResult:
    """Tests for ValidationResult formatting."""

    def test_format_errors_shows_all_errors(self):
        """format_errors displays all errors with location."""
        result = ValidationResult(
            errors=[
                ValidationError(
                    message="Error 1",
                    location="field1",
                    suggestion="Fix it this way",
                ),
                ValidationError(message="Error 2", location="field2"),
            ]
        )

        formatted = result.format_errors()

        assert "2 error(s)" in formatted
        assert "Error 1" in formatted
        assert "Error 2" in formatted
        assert "[field1]" in formatted
        assert "[field2]" in formatted
        assert "Fix it this way" in formatted

    def test_format_errors_includes_warnings(self):
        """format_errors includes warnings section."""
        result = ValidationResult(
            errors=[ValidationError(message="Error")],
            warnings=[ValidationWarning(message="Warning", location="somewhere")],
        )

        formatted = result.format_errors()

        assert "warning" in formatted.lower()
        assert "Warning" in formatted


class TestValidateConfigFunction:
    """Tests for top-level validate_config function."""

    def test_validate_config_with_valid_file(self):
        """validate_config returns valid result for good config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
target: test
max_depth: 2
intervals: [P7D]
entities:
  - alias: test
    table: test
    id: test_id
""")
            f.flush()

            result = validate_config(f.name)

            assert result.is_valid
            assert len(result.errors) == 0

            Path(f.name).unlink()

    def test_validate_config_with_invalid_file(self):
        """validate_config returns errors for bad config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
target: unknown
max_depth: 2
intervals: []
entities:
  - alias: test
    table: test
    id: id
""")
            f.flush()

            result = validate_config(f.name)

            assert not result.is_valid
            assert len(result.errors) > 0

            Path(f.name).unlink()


class TestFeaturizerIntegration:
    """Tests for Featurizer integration with validation."""

    def test_featurizer_uses_validation_by_default(self):
        """Featurizer runs validation by default."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
target: unknown_entity
max_depth: 2
intervals: []
entities:
  - alias: test
    table: test
    id: id
""")
            f.flush()

            from featurizer import Featurizer

            with pytest.raises(ValueError, match="validation failed"):
                Featurizer(f.name)

            Path(f.name).unlink()

    def test_featurizer_can_skip_validation(self):
        """Featurizer can skip enhanced validation if requested."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
target: test
max_depth: 2
intervals: [invalid_interval]
entities:
  - alias: test
    table: test
    id: id
""")
            f.flush()

            from featurizer import Featurizer

            # With validation=False, only basic checks run
            # This will pass basic validation but might fail later
            featurizer = Featurizer(f.name, validate=False)
            assert featurizer is not None

            Path(f.name).unlink()
