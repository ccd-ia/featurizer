# coding: utf-8

"""Configuration validation with detailed error reporting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import yaml


@dataclass
class ValidationError:
    """A single validation error with context."""

    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    line_number: Optional[int] = None

    def format(self) -> str:
        """Format error for display."""
        parts = []

        if self.location:
            parts.append(f"[{self.location}]")

        parts.append(self.message)

        if self.suggestion:
            parts.append(f"\n  → {self.suggestion}")

        return " ".join(parts)


@dataclass
class ValidationWarning:
    """A validation warning (non-fatal)."""

    message: str
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors)."""
        return len(self.errors) == 0

    def format_errors(self) -> str:
        """Format all errors for display."""
        if not self.errors:
            return "Configuration is valid."

        lines = [f"Configuration has {len(self.errors)} error(s):\n"]

        for i, error in enumerate(self.errors, 1):
            lines.append(f"  {i}. {error.format()}")

        if self.warnings:
            lines.append(f"\n{len(self.warnings)} warning(s):")
            for warning in self.warnings:
                location = f"[{warning.location}] " if warning.location else ""
                lines.append(f"  ⚠ {location}{warning.message}")

        return "\n".join(lines)


class ConfigValidator:
    """Multi-stage configuration validator."""

    VALID_VARIABLE_TYPES = {
        "numeric",
        "categorical",
        "text",
        "boolean",
        "date",
        "timestamp",
        "index",
        "vector",
    }

    ISO8601_DURATION_PATTERN = re.compile(
        r"^P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
    )

    def __init__(self, mode: str = "strict") -> None:
        """Initialize validator.

        Args:
            mode: Validation mode - "strict" (default) or "permissive"
        """
        self.mode = mode
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationWarning] = []

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Run all validation stages.

        Args:
            config: Configuration dictionary to validate

        Returns:
            ValidationResult with any errors and warnings
        """
        self.errors = []
        self.warnings = []

        # Stage 1: Structural validation
        self._validate_structure(config)
        if self.errors and self.mode == "strict":
            return ValidationResult(errors=self.errors, warnings=self.warnings)

        # Stage 2: Value validation
        self._validate_values(config)
        if self.errors and self.mode == "strict":
            return ValidationResult(errors=self.errors, warnings=self.warnings)

        # Stage 2b: Optional primitive selection
        self._validate_primitives(config)
        if self.errors and self.mode == "strict":
            return ValidationResult(errors=self.errors, warnings=self.warnings)

        # Stage 3: Semantic validation
        self._validate_semantics(config)

        # Stage 4: Best practices
        self._validate_best_practices(config)

        return ValidationResult(errors=self.errors, warnings=self.warnings)

    def _validate_structure(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure (required keys, types)."""
        required_keys = {"target", "max_depth", "intervals", "entities"}
        missing = [key for key in required_keys if key not in config]

        if missing:
            self.errors.append(
                ValidationError(
                    message=f"Missing required keys: {', '.join(missing)}",
                    suggestion="Required keys: target, max_depth, intervals, entities",
                )
            )
            return

        # Validate target
        if not isinstance(config["target"], str) or not config["target"].strip():
            self.errors.append(
                ValidationError(
                    message="'target' must be a non-empty string",
                    location="target",
                    suggestion="Specify the alias of the entity to generate features for",
                )
            )

        # Validate max_depth
        if not isinstance(config["max_depth"], int):
            self.errors.append(
                ValidationError(
                    message=f"'max_depth' must be an integer (got: {type(config['max_depth']).__name__})",
                    location="max_depth",
                    suggestion="Use an integer between 1 and 10 (recommended: 2-3)",
                )
            )
        elif config["max_depth"] < 1:
            self.errors.append(
                ValidationError(
                    message=f"'max_depth' must be positive (got: {config['max_depth']})",
                    location="max_depth",
                    suggestion="Use a value >= 1 (typical values: 2-3)",
                )
            )

        # Validate intervals
        if not isinstance(config["intervals"], list):
            self.errors.append(
                ValidationError(
                    message=f"'intervals' must be a list (got: {type(config['intervals']).__name__})",
                    location="intervals",
                    suggestion="Example: intervals: [P7D, P1M, P1Y]",
                )
            )

        # Validate entities
        if not isinstance(config["entities"], list):
            self.errors.append(
                ValidationError(
                    message=f"'entities' must be a list (got: {type(config['entities']).__name__})",
                    location="entities",
                )
            )
        elif not config["entities"]:
            self.errors.append(
                ValidationError(
                    message="'entities' list cannot be empty",
                    location="entities",
                    suggestion="Define at least one entity",
                )
            )

        # Validate relationships (if present)
        relationships = config.get("relationships")
        if relationships is not None and not isinstance(relationships, list):
            self.errors.append(
                ValidationError(
                    message=f"'relationships' must be a list (got: {type(relationships).__name__})",
                    location="relationships",
                )
            )

    def _validate_values(self, config: Dict[str, Any]) -> None:
        """Validate individual values (formats, ranges)."""
        # Validate intervals format
        if isinstance(config.get("intervals"), list):
            for i, interval in enumerate(config["intervals"]):
                if not isinstance(interval, str):
                    self.errors.append(
                        ValidationError(
                            message=f"Interval must be a string (got: {type(interval).__name__})",
                            location=f"intervals[{i}]",
                        )
                    )
                    continue

                if not self._is_valid_iso8601_duration(interval):
                    self.errors.append(
                        ValidationError(
                            message=f"Invalid ISO 8601 duration format: '{interval}'",
                            location=f"intervals[{i}]",
                            suggestion="Use format like P7D (7 days), P1W (1 week), P1M (1 month)",
                        )
                    )

        # Validate entities structure
        if isinstance(config.get("entities"), list):
            entity_aliases = set()
            for i, entity in enumerate(config["entities"]):
                if not isinstance(entity, dict):
                    self.errors.append(
                        ValidationError(
                            message=f"Entity must be a dictionary (got: {type(entity).__name__})",
                            location=f"entities[{i}]",
                        )
                    )
                    continue

                # Check required entity fields
                if "alias" not in entity:
                    self.errors.append(
                        ValidationError(
                            message="Entity missing required 'alias' field",
                            location=f"entities[{i}]",
                        )
                    )
                else:
                    alias = entity["alias"]
                    if alias in entity_aliases:
                        self.errors.append(
                            ValidationError(
                                message=f"Duplicate entity alias: '{alias}'",
                                location=f"entities[{i}].alias",
                                suggestion="Entity aliases must be unique",
                            )
                        )
                    entity_aliases.add(alias)

                if "table" not in entity:
                    self.errors.append(
                        ValidationError(
                            message="Entity missing required 'table' field",
                            location=f"entities[{i}]",
                        )
                    )

                # Validate variable types
                if "variables" in entity and isinstance(entity["variables"], dict):
                    for var_name, var_def in entity["variables"].items():
                        if not isinstance(var_def, dict) or "type" not in var_def:
                            self.errors.append(
                                ValidationError(
                                    message=f"Variable '{var_name}' missing 'type' field",
                                    location=f"entities[{i}].variables.{var_name}",
                                )
                            )
                            continue

                        var_type = var_def["type"]
                        if var_type not in self.VALID_VARIABLE_TYPES:
                            suggestion = self._suggest_similar(
                                var_type, self.VALID_VARIABLE_TYPES
                            )
                            self.errors.append(
                                ValidationError(
                                    message=f"Invalid variable type: '{var_type}'",
                                    location=f"entities[{i}].variables.{var_name}.type",
                                    suggestion=f"Valid types: {', '.join(sorted(self.VALID_VARIABLE_TYPES))}"
                                    + (
                                        f"\n  Did you mean '{suggestion}'?"
                                        if suggestion
                                        else ""
                                    ),
                                )
                            )

    def _validate_primitives(self, config: Dict[str, Any]) -> None:
        """Validate the optional `aggregations` / `transformations` selection.

        Either key may be omitted (the module defaults apply). When present it
        must be a list of registered primitive names; unknown names get a
        "Did you mean?" suggestion from the registry.
        """
        from .primitives.utils import list_aggregations, list_transformations

        checks = [
            ("aggregations", set(list_aggregations())),
            ("transformations", set(list_transformations())),
        ]
        for key, available in checks:
            value = config.get(key)
            if value is None:
                continue
            if not isinstance(value, list):
                self.errors.append(
                    ValidationError(
                        message=f"'{key}' must be a list of primitive names "
                        f"(got: {type(value).__name__})",
                        location=key,
                        suggestion=f"Example: {key}: [sum, mean]",
                    )
                )
                continue
            singular = key[:-1]  # "aggregation" / "transformation"
            for i, name in enumerate(value):
                if not isinstance(name, str):
                    self.errors.append(
                        ValidationError(
                            message=f"Primitive name must be a string "
                            f"(got: {type(name).__name__})",
                            location=f"{key}[{i}]",
                        )
                    )
                    continue
                if name not in available:
                    suggestion = self._suggest_similar(name, available)
                    self.errors.append(
                        ValidationError(
                            message=f"Unknown {singular} primitive: '{name}'",
                            location=f"{key}[{i}]",
                            suggestion=(
                                f"Did you mean '{suggestion}'?"
                                if suggestion
                                else "See: python -m featurizer list-primitives"
                            ),
                        )
                    )

    def _validate_semantics(self, config: Dict[str, Any]) -> None:
        """Validate semantic relationships between config parts."""
        entities = config.get("entities", [])
        relationships = config.get("relationships", [])
        target = config.get("target")

        if not isinstance(entities, list):
            return

        # Build entity lookup
        entity_aliases: set[str] = {
            str(e["alias"]) for e in entities if isinstance(e, dict) and "alias" in e
        }
        entity_map = {
            e["alias"]: e for e in entities if isinstance(e, dict) and "alias" in e
        }

        # Validate target exists
        if target and target not in entity_aliases:
            suggestion = self._suggest_similar(target, entity_aliases)
            self.errors.append(
                ValidationError(
                    message=f"Target entity '{target}' not found in entities",
                    location="target",
                    suggestion=f"Available entities: {', '.join(sorted(entity_aliases))}"
                    + (f"\n  Did you mean '{suggestion}'?" if suggestion else ""),
                )
            )

        # Validate relationships reference valid entities
        if isinstance(relationships, list):
            for i, rel in enumerate(relationships):
                if not isinstance(rel, dict):
                    continue

                parent = rel.get("parent", {})
                child = rel.get("child", {})

                if isinstance(parent, dict) and "entity" in parent:
                    parent_entity = parent["entity"]
                    if parent_entity not in entity_aliases:
                        suggestion = self._suggest_similar(
                            parent_entity, entity_aliases
                        )
                        self.errors.append(
                            ValidationError(
                                message=f"Relationship references unknown entity '{parent_entity}'",
                                location=f"relationships[{i}].parent.entity",
                                suggestion=f"Available entities: {', '.join(sorted(entity_aliases))}"
                                + (
                                    f"\n  Did you mean '{suggestion}'?"
                                    if suggestion
                                    else ""
                                ),
                            )
                        )

                if isinstance(child, dict) and "entity" in child:
                    child_entity = child["entity"]
                    if child_entity not in entity_aliases:
                        suggestion = self._suggest_similar(child_entity, entity_aliases)
                        self.errors.append(
                            ValidationError(
                                message=f"Relationship references unknown entity '{child_entity}'",
                                location=f"relationships[{i}].child.entity",
                                suggestion=f"Available entities: {', '.join(sorted(entity_aliases))}"
                                + (
                                    f"\n  Did you mean '{suggestion}'?"
                                    if suggestion
                                    else ""
                                ),
                            )
                        )

                # Validate temporal join requirements
                if "temporal" in rel and isinstance(rel["temporal"], dict):
                    temporal_mode = rel["temporal"].get("mode")
                    if temporal_mode == "as_of":
                        parent_entity = (
                            parent.get("entity") if isinstance(parent, dict) else None
                        )
                        child_entity = (
                            child.get("entity") if isinstance(child, dict) else None
                        )

                        missing_temporal = []
                        if parent_entity and parent_entity in entity_map:
                            if not entity_map[parent_entity].get("temporal_ix"):
                                missing_temporal.append(parent_entity)

                        if child_entity and child_entity in entity_map:
                            if not entity_map[child_entity].get("temporal_ix"):
                                missing_temporal.append(child_entity)

                        if missing_temporal:
                            self.warnings.append(
                                ValidationWarning(
                                    message=f"Temporal join (mode: as_of) requires temporal_ix on both entities. "
                                    f"Missing on: {', '.join(missing_temporal)}",
                                    location=f"relationships[{i}].temporal",
                                )
                            )

        # Detect circular relationships
        if isinstance(relationships, list) and len(relationships) > 1:
            cycles = self._detect_cycles(relationships, entity_aliases)
            for cycle in cycles:
                self.warnings.append(
                    ValidationWarning(
                        message=f"Circular relationship detected: {' -> '.join(cycle)}. "
                        f"This may cause deep traversal with high max_depth.",
                        location="relationships",
                    )
                )

    def _validate_best_practices(self, config: Dict[str, Any]) -> None:
        """Check for best practices and common issues."""
        # Warn about very high max_depth
        max_depth = config.get("max_depth")
        if isinstance(max_depth, int) and max_depth > 5:
            self.warnings.append(
                ValidationWarning(
                    message=f"max_depth={max_depth} is very high and may cause performance issues",
                    location="max_depth",
                )
            )

        # Warn about many intervals
        intervals = config.get("intervals", [])
        if isinstance(intervals, list) and len(intervals) > 10:
            self.warnings.append(
                ValidationWarning(
                    message=f"{len(intervals)} intervals will generate many features",
                    location="intervals",
                )
            )

    @staticmethod
    def _is_valid_iso8601_duration(duration: str) -> bool:
        """Check if string is valid ISO 8601 duration."""
        if not duration or duration == "P":
            return False
        return ConfigValidator.ISO8601_DURATION_PATTERN.match(duration) is not None

    @staticmethod
    def _suggest_similar(
        value: str, candidates: Set[str], max_distance: int = 2
    ) -> Optional[str]:
        """Suggest similar string from candidates using Levenshtein distance."""
        if not candidates:
            return None

        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        best_match = None
        best_distance = max_distance + 1

        for candidate in candidates:
            distance = levenshtein_distance(value.lower(), candidate.lower())
            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        return best_match if best_distance <= max_distance else None

    @staticmethod
    def _detect_cycles(
        relationships: List[Dict[str, Any]], entity_aliases: Set[str]
    ) -> List[List[str]]:
        """Detect circular relationships using DFS."""
        # Build adjacency list
        graph: Dict[str, List[str]] = {alias: [] for alias in entity_aliases}

        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            parent = rel.get("parent", {})
            child = rel.get("child", {})

            if isinstance(parent, dict) and isinstance(child, dict):
                parent_entity = parent.get("entity")
                child_entity = child.get("entity")

                if parent_entity and child_entity:
                    if parent_entity in graph and child_entity in graph:
                        graph[parent_entity].append(child_entity)

        # DFS to detect cycles
        def dfs(node: str, visited: Set[str], path: List[str]) -> Optional[List[str]]:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                cycle = dfs(neighbor, visited, path)
                if cycle:
                    return cycle

            path.pop()
            return None

        cycles = []
        visited: Set[str] = set()

        for entity in entity_aliases:
            if entity not in visited:
                cycle = dfs(entity, visited, [])
                if cycle:
                    cycles.append(cycle)

        return cycles


def validate_config(config_path: str, mode: str = "strict") -> ValidationResult:
    """Validate a configuration file.

    Args:
        config_path: Path to YAML configuration file
        mode: Validation mode - "strict" or "permissive"

    Returns:
        ValidationResult with any errors and warnings

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    validator = ConfigValidator(mode=mode)
    return validator.validate(config)
