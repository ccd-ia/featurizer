# coding: utf-8

"""Feature planning orchestration.

The planner traverses the entity graph, synthesizes features, and collects the
CTE definitions/joins that the SQL renderer expects.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Set

from icecream import ic
from loguru import logger

from .primitives import ERGraph, Entity, Feature, Relationship


@dataclass(frozen=True)
class PlannerResult:
    target: Entity
    features: Dict[str, Set[Feature]]
    joins: Dict[str, List[str]]
    ctes: List[str]


class FeaturePlanner:
    """Orchestrates feature traversal and aggregation synthesis."""

    def __init__(
        self,
        *,
        graph: ERGraph,
        target_alias: str,
        max_depth: int,
        intervals: Sequence[str],
        aggregations: Mapping[str, Callable],
        transformations: Mapping[str, Callable],
        debug: bool = False,
    ) -> None:
        self.graph = graph
        self.target_alias = target_alias
        self.max_depth = max_depth
        self.intervals = intervals
        self.aggregations = aggregations
        self.transformations = transformations
        self._debug_enabled = debug

        self._target: Entity | None = None
        self._features: Dict[str, Set[Feature]] = {}
        self._joins: Dict[str, List[str]] = {}
        self._ctes: List[str] = []
        self._path: List[Entity] = []

    def plan(self) -> PlannerResult:
        """Drive the DFS traversal and return the synthesized artifacts."""
        try:
            self._target = self.graph.entities[self.target_alias]
        except KeyError as exc:
            raise ValueError(f"Target entity '{self.target_alias}' not found in config.") from exc

        self._features = {entity.alias: set(entity.features) for entity in self.graph.entities.values()}
        self._joins = {entity.alias: [] for entity in self.graph.entities.values()}
        self._ctes = []
        self._path = []

        logger.debug("Starting feature build for target {}", self._target.alias)
        self._build_features(self._target)

        return PlannerResult(
            target=self._target,
            features={alias: set(features) for alias, features in self._features.items()},
            joins={alias: list(joins) for alias, joins in self._joins.items()},
            ctes=list(self._ctes),
        )

    # ------------------------------------------------------------------ #
    # Feature traversal helpers (ported from the original Featurizer)
    # ------------------------------------------------------------------ #

    def _build_features(self, target_entity: Entity, depth: int = 0) -> None:
        logger.debug("build_features({alias}) depth={depth}", alias=target_entity.alias, depth=depth)
        self._debug("build_features", entity=target_entity.alias, depth=depth)

        depth += 1
        if self.max_depth <= depth:
            logger.info("Maximum depth reached at depth {}", depth)
            return

        if target_entity not in self._path:
            self._path.append(target_entity)

        self._get_direct_features(target_entity, depth)
        self._get_backward_features(target_entity, depth)
        self._build_transformations(target_entity)

    @staticmethod
    def _sort_features(features: Iterable[Feature]) -> List[Feature]:
        return sorted(features, key=lambda feature: feature.name)

    def _get_direct_features(self, target: Entity, depth: int) -> None:
        forward_relationships = self.graph.get_forward_relationships(target)
        for relationship in forward_relationships:
            parent = relationship.parent
            if parent in self._path:
                continue
            self._build_features(parent, depth)
            self._build_direct(target, parent, relationship)

    def _get_backward_features(self, target: Entity, depth: int) -> None:
        backward_relationships = self.graph.get_backward_relationships(target)
        for relationship in backward_relationships:
            child = relationship.child
            if child in self._path:
                continue
            self._build_features(child, depth)
            self._build_aggregations(target, child, relationship)

    # ------------------------------------------------------------------ #
    # Aggregations / transformations / CTE assembly
    # ------------------------------------------------------------------ #

    def _build_aggregations(self, target: Entity, source: Entity, relationship: Relationship) -> None:
        logger.debug("Processing backward relationship {}", relationship)
        aggregations: List[Feature] = []

        for feature in self._features[source.alias]:
            for aggregator in self.aggregations.values():
                new_feature = aggregator(target, source, feature)
                if new_feature:
                    aggregations.append(new_feature)

                for interval in self.intervals:
                    if feature.entity.temporal_ix is None:
                        logger.warning(
                            "Entity {} lacks temporal index; skipping interval-based aggregation",
                            feature.entity,
                        )
                        break
                    interval_feature = aggregator(target, source, feature, interval=interval)
                    if interval_feature:
                        aggregations.append(interval_feature)

        aggregation_set = set(aggregations)
        sorted_aggs = self._sort_features(aggregation_set)
        self._debug("aggregations", target=target.alias, source=source.alias, count=len(sorted_aggs))

        self._features[source.alias].update(aggregation_set)
        self._features[target.alias].update(aggregation_set)

        self._build_aggregations_cte(target, source, relationship, sorted_aggs)

    def _build_direct(self, target: Entity, source: Entity, relationship: Relationship) -> None:
        logger.debug("Processing forward relationship {}", relationship)
        directs = self._sort_features(self._features[source.alias])
        self._debug("direct_features", target=target.alias, source=source.alias, count=len(directs))

        self._features[target.alias].update(directs)
        if getattr(relationship, "temporal_mode", None) == "as_of":
            self._build_direct_asof(target, source, relationship, directs)
        else:
            self._build_direct_cte(target, source, relationship, directs)

    def _build_transformations(self, target: Entity) -> None:
        self._build_synth_cte(target)

        transformed: List[Feature | Iterable[Feature]] = []
        for feature in self._features[target.alias]:
            if feature.type != 'index':
                for transformer in self.transformations.values():
                    new_feature = transformer(target, feature)
                    if new_feature:
                        transformed.append(new_feature)
            else:
                transformed.append(feature)

        flattened: Set[Feature] = set()
        for candidate in transformed:
            if isinstance(candidate, (list, tuple, set)):
                flattened.update(candidate)
            elif candidate:
                flattened.add(candidate)

        self._debug("transformations", target=target.alias, count=len(flattened))
        self._features[target.alias].update(flattened)
        sorted_flattened = self._sort_features(flattened)
        self._build_transform_cte(target, sorted_flattened)

    # ------------------------------------------------------------------ #
    # CTE builders (unchanged from the original implementation)
    # ------------------------------------------------------------------ #

    def _build_aggregations_cte(
        self,
        target: Entity,
        source: Entity,
        relationship: Relationship,
        features: Iterable[Feature],
    ) -> None:
        cte_name = f"{source.alias}_aggs_for_{target.alias}"
        join_statement = (
            f" {cte_name} on {cte_name}.{relationship.child_key} = "
            f"{relationship.parent.table}.{relationship.parent_key} "
        )

        rendered_features = [feature.query for feature in features if feature.type not in ['key']]
        where_clause = (
            f"where aod.as_of_date >= {source.temporal_ix.name}" if source.temporal_ix else ''
        )

        cte_query = f"""
        -- Aggregate for {target.alias}
        {cte_name} as (
        select
        {source.alias}_transform.{relationship.parent_key},
        {','.join(rendered_features)}
        from {source.alias}_transform
        {where_clause}
        group by {relationship.parent_key}
        )
        """
        self._joins[target.alias].append(join_statement)
        self._ctes.append(cte_query)

    def _build_direct_cte(
        self,
        target: Entity,
        source: Entity,
        relationship: Relationship,
        features: Iterable[Feature],
    ) -> None:
        if source.id is None:
            logger.debug("Skipping direct features for {} because it lacks an id column.", source.alias)
            return

        cte_name = f"{source.alias}_direct_transfers_for_{target.alias}"

        feature_names = [feature.name for feature in features if feature.type not in ['index', 'key']]
        cte_query = f"""
        -- direct features for {target.alias}
        {cte_name} as (
        select
        {source.id.name},
        {','.join(feature_names)}
        from {source.alias}_transform
        )
        """
        join_statement = (
            f" {cte_name} on {cte_name}.{relationship.child_key} = "
            f"{relationship.child.table}.{relationship.child_key} "
        )
        self._joins[target.alias].append(join_statement)
        self._ctes.append(cte_query)

    def _build_direct_asof(
        self,
        target: Entity,
        source: Entity,
        relationship: Relationship,
        features: Iterable[Feature],
    ) -> None:
        target_temporal = target.temporal_ix.name if target.temporal_ix else None
        source_temporal = (
            relationship.temporal_child_field
            or (source.temporal_ix.name if source.temporal_ix else None)
        )
        if not target_temporal or not source_temporal:
            logger.warning(
                "Temporal join requested between {} and {} but temporal indexes are missing; falling back to static join.",
                source.alias,
                target.alias,
            )
            self._build_direct_cte(target, source, relationship, features)
            return

        projected = [
            f"{source.alias}_transform.{feature.name} as \"{feature.name}\""
            for feature in features
            if feature.type not in {'index', 'key'}
        ]
        if not projected:
            return

        projected_sql = ",\n        ".join(projected)

        where_clauses = [
            f"{source.alias}_transform.{relationship.parent_key} = {target.table}.{relationship.child_key}",
            f"{source.alias}_transform.{source_temporal} <= {target.table}.{target_temporal}",
        ]
        if relationship.temporal_grace:
            where_clauses.append(
                f"{target.table}.{target_temporal} - {source.alias}_transform.{source_temporal} <= interval '{relationship.temporal_grace}'"
            )

        cte_name = f"{source.alias}_asof_for_{target.alias}"
        # Convert the CTE text into a lateral join referencing the target table row.
        lateral_join = (
            " lateral (\n"
            "        select\n"
            f"        {projected_sql}\n"
            f"        from {source.alias}_transform\n"
            f"        where {' and '.join(where_clauses)}\n"
            f"        order by {source.alias}_transform.{source_temporal} desc\n"
            "        limit 1\n"
            f"    ) as {cte_name} on true "
        )
        self._joins[target.alias].append(lateral_join)

    def _build_synth_cte(self, target: Entity) -> None:
        cte_table = f"{target.alias}_synth"

        indexes = [f"{target.table}.{ix.name}" for ix in target.indexes]
        keys = [f"{target.table}.{key.name}" for key in target.keys]
        feature_names = [
            feature.name
            for feature in self._sort_features(self._features[target.alias])
            if feature.type not in ['index', 'key']
        ]

        cte_query = f"""
        -- sythetize aggregations and direct features for {target.alias}
        {cte_table} as (
        select
        {', '.join(indexes + keys + feature_names)}
        from {target.table}
        {' left join ' if self._joins[target.alias] else '' }
        {' left join '.join(self._joins[target.alias])}
        )
        """

        self._ctes.append(cte_query)

    def _build_transform_cte(self, target: Entity, features: Iterable[Feature]) -> None:
        cte_table = f"{target.alias}_transform"

        indexes = [f"{ix.name}" for ix in target.indexes]
        keys = [f"{key.name}" for key in target.keys]
        rendered_features = [feature.query for feature in features if feature.type not in ['index', 'key']]

        cte_query = f"""
        -- transform {target.alias}
        {cte_table} as (
        select
        {', '.join(indexes + keys + rendered_features)}
        from {target.alias}_synth
        )
        """

        self._ctes.append(cte_query)

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #

    def _debug(self, message: str, **context) -> None:
        if not self._debug_enabled:
            return
        payload = {"message": message}
        if context:
            payload.update(context)
        ic(payload)
