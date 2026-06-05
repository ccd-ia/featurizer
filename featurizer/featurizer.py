# coding: utf-8

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Set

import pandas as pd
import yaml
from icecream import ic
from loguru import logger

from .executor import QueryExecutor
from .planner import FeaturePlanner, PlannerResult
from .primitives import Entity, ERGraph, Feature
from .primitives.utils import (
    AggregationRegistry,
    TransformationRegistry,
    get_aggregations,
    get_transformers,
)
from .sql import SQLRenderer
from .validation import ConfigValidator

DEFAULT_AGGREGATIONS = (
    "count",
    "mean",
    "sum",
    "stddev",
    "min",
    "max",
    "median",
    "nunique",
    "recency",
    "tenure",
)
DEFAULT_TRANSFORMATIONS = (
    "identity",
    "abs",
    "cum_sum",
    "day",
    "dow",
    "month",
    "lag_1",
    "lag_3",
    "lag_7",
    "rolling_mean_3",
    "rolling_std_7",
    "rolling_median_7",
    "rolling_iqr_7",
    "ema_7",
    "holt_winters_level_7",
    "holt_winters_trend_7",
    "pct_change_1",
)


class Featurizer:
    """
    PostgreSQL implementation of the DFS algorithm (adapted for temporal data sets).

    Coordinates configuration loading, feature planning, SQL rendering, and optional
    database execution.
    """

    def __init__(
        self, config_file: str, *, debug: bool = False, validate: bool = True
    ) -> None:
        """Initialize Featurizer from a YAML configuration file.

        Args:
            config_file: Path to YAML configuration file
            debug: Enable debug logging with icecream. Can also be set via FEATURIZER_DEBUG env var.
            validate: Run enhanced validation checks (default: True)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is missing required keys or has invalid values
        """
        config = self._load_config(config_file, validate=validate)

        self._debug_enabled: bool = debug or self._env_debug_enabled()
        if self._debug_enabled:
            ic.configureOutput(prefix="[Featurizer] ", includeContext=True)

        self.max_depth: int = config["max_depth"]
        self.intervals: List[str] = config["intervals"]

        self.graph: ERGraph = ERGraph(config["entities"], config["relationships"])
        self.target: Entity = self._get_entity(config["target"])

        # Primitive selection: config may override the active set; otherwise the
        # curated module defaults apply. Unknown names raise in get_* (and are
        # caught earlier with suggestions by the validator when validate=True).
        agg_names = config.get("aggregations") or DEFAULT_AGGREGATIONS
        tx_names = config.get("transformations") or DEFAULT_TRANSFORMATIONS
        self.aggregations: AggregationRegistry = get_aggregations(agg_names)
        self.transformations: TransformationRegistry = get_transformers(tx_names)

        planner = FeaturePlanner(
            graph=self.graph,
            target_alias=self.target.alias,
            max_depth=self.max_depth,
            intervals=self.intervals,
            aggregations=self.aggregations,
            transformations=self.transformations,
            debug=self._debug_enabled,
        )
        self._plan: PlannerResult = planner.plan()

        self.features: Dict[str, Set[Feature]] = {
            alias: set(features) for alias, features in self._plan.features.items()
        }
        self.ctes: List[str] = list(self._plan.ctes)
        self.joins: Dict[str, List[str]] = {
            alias: list(joins) for alias, joins in self._plan.joins.items()
        }

        self._renderer: SQLRenderer = SQLRenderer()
        self._executor: QueryExecutor = QueryExecutor()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def entities(self) -> Iterable[Entity]:
        """Return all entities in the graph."""
        return self.graph.entities.values()

    @property
    def relationships(self) -> List[Any]:
        """Return all relationships in the graph."""
        return self.graph.relationships

    @property
    def query(self) -> str:
        """Generate the SQL query for this featurizer configuration."""
        return self._renderer.render(self._plan)

    def to_dataframe(
        self, *, impute: bool = False, **impute_kwargs: Any
    ) -> pd.DataFrame:
        """Execute the query and return results as a DataFrame.

        Args:
            impute: When True, run the opt-in imputation pass (count-like
                features → 0, measures left NULL unless ``measure_strategy`` is
                given, with ``<feature>__missing`` indicator columns). The
                default keeps the raw NULLs, since missingness is signal.
            **impute_kwargs: Forwarded to
                :func:`featurizer.imputation.impute_features`.

        Returns:
            DataFrame indexed by ['as_of_date', target_id]

        Raises:
            ValueError: If target entity doesn't define a primary ID
        """
        if self.target.id is None:
            raise ValueError(
                f"Target entity '{self.target.alias}' does not define a primary id."
            )
        df = self._executor.to_dataframe(self.query, self.target.id.name)
        if impute:
            from .imputation import impute_features

            df = impute_features(df, **impute_kwargs)
        return df

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_entity(self, alias: str) -> Entity:
        """Get an entity by alias from the graph.

        Args:
            alias: Entity alias to look up

        Returns:
            Entity with the given alias

        Raises:
            ValueError: If entity with alias doesn't exist
        """
        entity = self.graph.entities.get(alias)
        if entity is None:
            raise ValueError(f"Unknown target entity alias '{alias}'.")
        return entity

    @staticmethod
    def _env_debug_enabled() -> bool:
        """Check if debug mode is enabled via environment variable."""
        value = os.getenv("FEATURIZER_DEBUG", "")
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_config(config_file: str, validate: bool = True) -> Dict[str, Any]:
        """Load and validate configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file
            validate: Run enhanced validation checks

        Returns:
            Validated configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or missing required keys
        """
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config file not found: {config_file}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in config file {config_file}") from exc

        # Run enhanced validation if enabled
        if validate:
            validator = ConfigValidator(mode="strict")
            result = validator.validate(config)

            if not result.is_valid:
                raise ValueError(
                    f"Configuration validation failed:\n{result.format_errors()}"
                )

            # Log warnings
            for warning in result.warnings:
                location = f"[{warning.location}] " if warning.location else ""
                logger.warning(f"{location}{warning.message}")

        # Backwards compatibility: Basic validation
        required_keys = {"target", "max_depth", "intervals", "entities"}
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Config missing required keys: {', '.join(missing)}")

        if not isinstance(config["target"], str) or not config["target"].strip():
            raise ValueError("'target' must be a non-empty string.")

        if not isinstance(config["max_depth"], int) or config["max_depth"] < 1:
            raise ValueError("'max_depth' must be a positive integer.")

        if not isinstance(config["entities"], list) or not config["entities"]:
            raise ValueError("Config must declare at least one entity in 'entities'.")

        if not isinstance(config["intervals"], list):
            raise ValueError("'intervals' must be a list of interval strings.")

        relationships = config.get("relationships")
        if relationships is None:
            logger.debug(
                "No relationships defined in config; defaulting to empty list."
            )
            config["relationships"] = []
        elif not isinstance(relationships, list):
            raise ValueError("'relationships' must be a list when provided.")

        return config
