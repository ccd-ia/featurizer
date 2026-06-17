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

        self.graph: ERGraph = ERGraph(
            config["entities"],
            config["relationships"],
            config.get("spatial_relationships"),
        )
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
            from .imputation import guard_full_matrix_fit, impute_features

            # Engine path: this fits over the whole returned matrix, so gate the
            # leaky measure strategies (ADR-0001). The pure impute_features helper
            # stays ungated for callers that pre-split their own data.
            guard_full_matrix_fit(
                impute_kwargs.get("measure_strategy", "none"),
                allow_full_matrix_fit=bool(
                    impute_kwargs.pop("allow_full_matrix_fit", False)
                ),
                caller="to_dataframe",
            )
            df = impute_features(df, **impute_kwargs)
        return df

    def to_arrow(
        self,
        *,
        connection: Any = None,
        numeric_as_float: bool = True,
        impute: bool = False,
        **impute_kwargs: Any,
    ) -> "Any":
        """Execute the query and return a :class:`pyarrow.Table`, no pandas hop.

        Streams the result out of PostgreSQL with binary ``COPY`` and decodes it
        column-by-column into Arrow, so SQL NULLs are preserved as Arrow nulls
        (never coerced to ``NaN``) and the full result set never materializes as
        a pandas frame. ``as_of_date`` and the target id are ordinary leading
        columns (no index), unlike :meth:`to_dataframe`.

        Args:
            connection: An open psycopg connection to run ``COPY`` on. Required
                when the rendered query references session ``TEMP`` tables (the
                integration harness). When ``None``, a connection is built from
                ``DATABASE_URL`` / ``PG*`` and closed afterwards.
            numeric_as_float: Cast PostgreSQL ``numeric`` aggregate columns to
                ``float64`` (ML-ready, ``to_dataframe``-comparable). Set ``False``
                to keep exact ``decimal128``. NULLs are preserved either way.
            impute: When True, apply the Arrow-native imputation pass
                (:func:`featurizer.imputation.impute_arrow`): count-like features
                → 0, measures left null unless ``measure_strategy`` is given, with
                stable ``<feature>__missing`` indicator columns. ``as_of_date``
                and the target id are passed as ``key_columns`` and left untouched.
            **impute_kwargs: Forwarded to ``impute_arrow``. ``measure_strategy`` in
                ``{"mean","median"}`` additionally requires
                ``allow_full_matrix_fit=True`` (ADR-0001 leakage gate).

        Returns:
            A pyarrow.Table.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
            ValueError: If the target entity does not define a primary id, or a
                leaky measure strategy is requested without the opt-in.
        """
        if self.target.id is None:
            raise ValueError(
                f"Target entity '{self.target.alias}' does not define a primary id."
            )
        from .arrow import ArrowExporter

        table = ArrowExporter().to_arrow(
            self.query,
            connection=connection,
            numeric_as_float=numeric_as_float,
        )
        if impute:
            from .imputation import guard_full_matrix_fit, impute_arrow

            guard_full_matrix_fit(
                impute_kwargs.get("measure_strategy", "none"),
                allow_full_matrix_fit=bool(
                    impute_kwargs.pop("allow_full_matrix_fit", False)
                ),
                caller="to_arrow",
            )
            table = impute_arrow(
                table,
                key_columns=("as_of_date", self.target.id.name),
                **impute_kwargs,
            )
        return table

    def to_parquet(
        self,
        path: str,
        *,
        connection: Any = None,
        numeric_as_float: bool = True,
        impute: bool = False,
        **impute_kwargs: Any,
    ) -> None:
        """Execute the query and write the result to a Parquet file.

        Thin wrapper over :meth:`to_arrow` plus ``pyarrow.parquet.write_table``;
        all arguments (including the imputation contract and its ADR-0001 leakage
        gate) behave exactly as in :meth:`to_arrow`. NULLs are written as Parquet
        nulls.

        Args:
            path: Destination ``.parquet`` file path.
            connection: See :meth:`to_arrow`.
            numeric_as_float: See :meth:`to_arrow`.
            impute: See :meth:`to_arrow`.
            **impute_kwargs: See :meth:`to_arrow`.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
        """
        table = self.to_arrow(
            connection=connection,
            numeric_as_float=numeric_as_float,
            impute=impute,
            **impute_kwargs,
        )
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        pq.write_table(table, path)

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
