"""Tests for enhanced configuration validation."""

import tempfile
from pathlib import Path

import pytest

from featurizer import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    validate_config,
)
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

    def test_as_of_boundary_inclusive_passes(self):
        """The default-equivalent 'inclusive' boundary is accepted."""
        config = {
            "target": "users",
            "max_depth": 2,
            "intervals": [],
            "as_of_boundary": "inclusive",
            "entities": [{"alias": "users", "table": "users", "id": "user_id"}],
        }

        result = ConfigValidator().validate(config)

        assert result.is_valid

    def test_as_of_boundary_exclusive_passes(self):
        """The 'exclusive' boundary is accepted."""
        config = {
            "target": "users",
            "max_depth": 2,
            "intervals": [],
            "as_of_boundary": "exclusive",
            "entities": [{"alias": "users", "table": "users", "id": "user_id"}],
        }

        result = ConfigValidator().validate(config)

        assert result.is_valid

    def test_invalid_as_of_boundary_generates_error(self):
        """An unknown boundary value is rejected with a suggestion."""
        config = {
            "target": "users",
            "max_depth": 2,
            "intervals": [],
            "as_of_boundary": "inclsive",  # typo
            "entities": [{"alias": "users", "table": "users", "id": "user_id"}],
        }

        result = ConfigValidator().validate(config)

        assert not result.is_valid
        assert any(
            "as_of_boundary" in (error.location or "") for error in result.errors
        )
        assert any("inclusive" in (error.suggestion or "") for error in result.errors)

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
            "vector",
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

    def test_unknown_temporal_key_warns_with_suggestion(self):
        """Unknown temporal-block keys warn; child_time suggests child_timestamp."""
        config = {
            "target": "patients",
            "max_depth": 2,
            "intervals": [],
            "entities": [
                {
                    "alias": "patients",
                    "table": "patients",
                    "id": "patient_id",
                    "temporal_ix": "admission_date",
                },
                {
                    "alias": "care_plans",
                    "table": "care_plans",
                    "id": "plan_id",
                    "temporal_ix": "plan_date",
                },
            ],
            "relationships": [
                {
                    "parent": {"entity": "care_plans", "key": "patient_id"},
                    "child": {"entity": "patients", "key": "patient_id"},
                    "temporal": {"mode": "as_of", "child_time": "plan_date"},
                }
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid  # warning, not error
        unknown = [
            w for w in result.warnings if "Unknown key 'child_time'" in w.message
        ]
        assert len(unknown) == 1
        assert "Did you mean 'child_timestamp'?" in unknown[0].message
        assert unknown[0].location == "relationships[0].temporal.child_time"

    def test_known_temporal_keys_do_not_warn(self):
        """mode/grace/child_timestamp are the parsed keys; no unknown-key warning."""
        config = {
            "target": "patients",
            "max_depth": 2,
            "intervals": [],
            "entities": [
                {
                    "alias": "patients",
                    "table": "patients",
                    "id": "patient_id",
                    "temporal_ix": "admission_date",
                },
                {
                    "alias": "care_plans",
                    "table": "care_plans",
                    "id": "plan_id",
                    "temporal_ix": "plan_date",
                },
            ],
            "relationships": [
                {
                    "parent": {"entity": "care_plans", "key": "patient_id"},
                    "child": {"entity": "patients", "key": "patient_id"},
                    "temporal": {
                        "mode": "as_of",
                        "grace": "P7D",
                        "child_timestamp": "plan_date",
                    },
                }
            ],
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid
        assert not any("Unknown key" in w.message for w in result.warnings)


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
                {
                    "parent": {"entity": "A", "key": "id"},
                    "child": {"entity": "B", "key": "a_id"},
                },
                {
                    "parent": {"entity": "B", "key": "id"},
                    "child": {"entity": "A", "key": "b_id"},
                },
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
        assert any(
            "max_depth" in warning.message.lower() for warning in result.warnings
        )

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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


class TestEmptyMatrixWarning:
    """validate_config warns when the planned matrix has no feature column.

    A window transformer needs a timeline; since 1.3.1 (issue #76) it is not
    rendered on an entity without ``temporal_ix``. A transformer list made only
    of such windows, on an undated target, used to validate clean and plan a
    query that selects the id alone.
    """

    REPRO = """
target: patients
max_depth: 1
intervals: [P1D]
transformations: [lag_1]
entities:
  - alias: patients
    id: patient_id
    table: analytics.patients
    variables:
      age: {type: numeric}
"""

    @staticmethod
    def _write(tmp_path: Path, text: str) -> str:
        path = tmp_path / "config.yaml"
        path.write_text(text)
        return str(path)

    @staticmethod
    def _empty_matrix_warnings(result: ValidationResult) -> list:
        return [w for w in result.warnings if "zero feature columns" in w.message]

    def test_undated_target_with_only_window_transformers_warns(self, tmp_path):
        """The issue's repro: one warning naming entity, cause and fixes."""
        result = validate_config(self._write(tmp_path, self.REPRO))

        assert result.is_valid
        warnings = self._empty_matrix_warnings(result)
        assert len(warnings) == 1
        message = warnings[0].message
        assert "'patients'" in message
        assert "no temporal_ix" in message
        assert "lag_1" in message
        assert "'identity'" in message
        assert "'temporal_ix'" in message
        assert warnings[0].location == "transformations"

    def test_cli_prints_the_warning_and_exits_zero(self, tmp_path, capsys):
        """The validate verb shows the warning; the exit code stays 0."""
        import argparse

        from featurizer.cli import validate_command

        path = self._write(tmp_path, self.REPRO)
        assert validate_command(argparse.Namespace(config=path)) == 0
        out = capsys.readouterr().out
        assert "Configuration is valid" in out
        assert "zero feature columns" in out

    def test_identity_in_the_list_is_silent(self, tmp_path):
        """identity keeps the raw variable, so the matrix is not empty."""
        text = self.REPRO.replace("[lag_1]", "[identity, lag_1]")
        result = validate_config(self._write(tmp_path, text))

        assert result.is_valid
        assert self._empty_matrix_warnings(result) == []

    def test_dated_target_is_silent(self, tmp_path):
        """With a timeline lag_1 renders, so the matrix has a column."""
        text = self.REPRO.replace(
            "    table: analytics.patients\n",
            "    table: analytics.patients\n    temporal_ix: admitted_at\n",
        )
        result = validate_config(self._write(tmp_path, text))

        assert result.is_valid
        assert self._empty_matrix_warnings(result) == []

    CHILD = """
  - alias: visits
    id: visit_id
    table: analytics.visits
    temporal_ix: visited_at
    variables:
      cost: {type: numeric}
      patient_id: {type: index}
relationships:
  - parent: {entity: patients, key: patient_id}
    child: {entity: visits, key: patient_id}
aggregations: [count, mean]
"""

    def test_undated_target_with_aggregated_child_is_silent(self, tmp_path):
        """Aggregates over a dated child are columns of the undated target."""
        text = (
            self.REPRO.replace("max_depth: 1", "max_depth: 2")
            .replace("transformations: [lag_1]\n", "")
            .replace(
                "      age: {type: numeric}\n", "      patient_id: {type: index}\n"
            )
            + self.CHILD
        )
        path = self._write(tmp_path, text)
        result = validate_config(path)

        assert result.is_valid
        assert self._empty_matrix_warnings(result) == []

        from featurizer import Featurizer

        assert Featurizer(path).feature_manifest  # the control: columns exist

    def test_window_only_list_drops_the_aggregates_too(self, tmp_path):
        """The target's transformers also carry its aggregates to the output,
        so ``[lag_1]`` on an undated target empties it even with a child."""
        text = self.REPRO.replace("max_depth: 1", "max_depth: 2") + self.CHILD
        result = validate_config(self._write(tmp_path, text))

        assert result.is_valid
        warnings = self._empty_matrix_warnings(result)
        assert len(warnings) == 1
        assert "lag_1" in warnings[0].message

    def test_categorical_without_vocabulary_is_not_planned(self, tmp_path, monkeypatch):
        """An undeclared vocabulary would introspect the DB: skip the check."""
        import featurizer.arrow as arrow

        def _no_database():
            raise AssertionError("validate_config opened a database connection")

        monkeypatch.setattr(arrow, "default_connection", _no_database)
        text = self.REPRO + "      sex: {type: categorical, role: categorical}\n"
        result = validate_config(self._write(tmp_path, text))

        assert result.is_valid
        assert self._empty_matrix_warnings(result) == []

    def test_planning_failure_does_not_raise(self, tmp_path, monkeypatch):
        """A planner error leaves validation as it was: no warning, no raise."""
        import featurizer.featurizer as featurizer_module

        def _boom(*args, **kwargs):
            raise RuntimeError("planner failed")

        from loguru import logger

        monkeypatch.setattr(featurizer_module, "Featurizer", _boom)
        messages: list = []
        sink = logger.add(messages.append, level="DEBUG", format="{message}")
        try:
            result = validate_config(self._write(tmp_path, self.REPRO))
        finally:
            logger.remove(sink)

        assert result.is_valid
        assert self._empty_matrix_warnings(result) == []
        # Not swallowed silently: the skip is logged with its traceback.
        logged = [m for m in messages if "empty-matrix check skipped" in m]
        assert len(logged) == 1
        assert logged[0].record["exception"] is not None
