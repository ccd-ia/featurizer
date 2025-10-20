# coding: utf-8

import os
import yaml
from icecream import ic
from loguru import logger

from .executor import QueryExecutor
from .planner import FeaturePlanner, PlannerResult
from .primitives import ERGraph
from .primitives.utils import get_aggregations, get_transformers
from .sql import SQLRenderer

DEFAULT_AGGREGATIONS = ("count", "mean", "sum", "stddev")
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

    def __init__(self, config_file, *, debug=False):
        config = self._load_config(config_file)

        self._debug_enabled = debug or self._env_debug_enabled()
        if self._debug_enabled:
            ic.configureOutput(prefix="[Featurizer] ", includeContext=True)

        self.max_depth = config["max_depth"]
        self.intervals = config["intervals"]

        self.graph = ERGraph(config["entities"], config["relationships"])
        self.target = self._get_entity(config["target"])

        self.aggregations = get_aggregations(DEFAULT_AGGREGATIONS)
        self.transformations = get_transformers(DEFAULT_TRANSFORMATIONS)

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

        self.features = {alias: set(features) for alias, features in self._plan.features.items()}
        self.ctes = list(self._plan.ctes)
        self.joins = {alias: list(joins) for alias, joins in self._plan.joins.items()}

        self._renderer = SQLRenderer()
        self._executor = QueryExecutor()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def entities(self):
        return self.graph.entities.values()

    @property
    def relationships(self):
        return self.graph.relationships

    @property
    def query(self):
        return self._renderer.render(self._plan)

    def to_dataframe(self):
        if self.target.id is None:
            raise ValueError(f"Target entity '{self.target.alias}' does not define a primary id.")
        return self._executor.to_dataframe(self.query, self.target.id.name)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_entity(self, alias):
        entity = self.graph.entities.get(alias)
        if entity is None:
            raise ValueError(f"Unknown target entity alias '{alias}'.")
        return entity

    @staticmethod
    def _env_debug_enabled():
        value = os.getenv("FEATURIZER_DEBUG", "")
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_config(config_file):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config file not found: {config_file}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in config file {config_file}") from exc

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
            logger.debug("No relationships defined in config; defaulting to empty list.")
            config["relationships"] = []
        elif not isinstance(relationships, list):
            raise ValueError("'relationships' must be a list when provided.")

        return config
