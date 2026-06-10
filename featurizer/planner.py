# coding: utf-8

"""Feature planning orchestration.

The planner traverses the entity graph, synthesizes features, and collects the
CTE definitions/joins that the SQL renderer expects.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Set

from icecream import ic
from loguru import logger

from .primitives import Entity, ERGraph, Feature, Relationship


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
        # Names of the columns each <alias>_synth CTE projects. The transform
        # CTE reads from synth, so any feature already materialized there must
        # be referenced by name rather than re-rendering its definition.
        self._synth_columns: Dict[str, Set[str]] = {}

    def plan(self) -> PlannerResult:
        """Drive the DFS traversal and return the synthesized artifacts."""
        try:
            self._target = self.graph.entities[self.target_alias]
        except KeyError as exc:
            raise ValueError(
                f"Target entity '{self.target_alias}' not found in config."
            ) from exc

        self._features = {
            entity.alias: set(entity.features)
            for entity in self.graph.entities.values()
        }
        self._joins = {entity.alias: [] for entity in self.graph.entities.values()}
        self._ctes = []
        self._path = []
        self._synth_columns = {}

        logger.debug("Starting feature build for target {}", self._target.alias)
        self._build_features(self._target)

        return PlannerResult(
            target=self._target,
            features={
                alias: set(features) for alias, features in self._features.items()
            },
            joins={alias: list(joins) for alias, joins in self._joins.items()},
            ctes=list(self._ctes),
        )

    # ------------------------------------------------------------------ #
    # Feature traversal helpers (ported from the original Featurizer)
    # ------------------------------------------------------------------ #

    def _build_features(self, target_entity: Entity, depth: int = 0) -> None:
        logger.debug(
            "build_features({alias}) depth={depth}",
            alias=target_entity.alias,
            depth=depth,
        )
        self._debug("build_features", entity=target_entity.alias, depth=depth)

        depth += 1
        if target_entity not in self._path:
            self._path.append(target_entity)

        # Depth bounds how deep we recurse into neighbours, but every entity we
        # actually reach must still be materialized: a parent's aggregation CTE
        # reads ``from <child>_transform``, and the final query selects
        # ``from <target>_transform``. Returning before _build_transformations
        # (the previous behaviour) left those CTEs undefined -> invalid SQL.
        if self.max_depth > depth:
            self._get_direct_features(target_entity, depth)
            self._get_backward_features(target_entity, depth)
        else:
            logger.info(
                "Maximum recursion depth reached at depth {}; materializing {} "
                "without traversing further.",
                depth,
                target_entity.alias,
            )

        self._build_graph_features(target_entity)
        self._build_transformations(target_entity)

    @staticmethod
    def _sort_features(features: Iterable[Feature]) -> List[Feature]:
        return sorted(features, key=lambda feature: feature.name)

    @staticmethod
    def _carried_index_columns(target: Entity) -> List[str]:
        """Index-typed *variables* that must be projected but are not features.

        A foreign key declared ``type: index`` (e.g. ``care_plans.patient_id``)
        is neither the entity's own id/temporal/spatial index nor a registered
        relationship key, so it falls out of the normal projection. It is still
        needed as a column — notably for the as-of join's ``WHERE`` clause — so
        carry it through synth/transform by name.
        """
        covered = {ix.name for ix in target.indexes} | {key.name for key in target.keys}
        return sorted(
            {
                feature.name
                for feature in target.features
                if feature.type == "index" and feature.name not in covered
            }
        )

    @classmethod
    def _identifier_columns(cls, target: Entity) -> List[str]:
        """Distinct identifier column names to project from the base table.

        Combines id/temporal/spatial indexes, relationship keys, and carried
        index variables. The same name can appear as both the entity id and a
        relationship key when a primary key doubles as a foreign key (an entity
        keyed by ``patient_id`` that is also the child of a ``patient_id``
        relationship). Projecting it twice makes the reference ambiguous, so
        dedupe by name while preserving order.
        """
        names = (
            [ix.name for ix in target.indexes]
            + [key.name for key in target.keys]
            + cls._carried_index_columns(target)
        )
        seen: Set[str] = set()
        ordered: List[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

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

    def _build_graph_features(self, node: Entity) -> None:
        """Attach graph features (degree family) for every edge on this node."""
        for edge in self.graph.get_edges_for_node(node):
            self._build_graph_cte(node, edge)

    def _build_graph_cte(self, node: Entity, edge) -> None:
        """Emit a degree CTE for ``node`` over the edges in ``edge`` and join it.

        Degree is computed by unioning each edge as an outgoing row for its
        source node and an incoming row for its target node, then grouping by
        node id. When the edge carries a ``timestamp`` the union is bounded by
        ``<= aod.as_of_date`` so degree is measured as-of each cutoff (the same
        causal guarantee the aggregation CTEs use); without one the graph is
        treated as static and leakage is the caller's responsibility.
        """
        if node.id is None:
            logger.warning(
                "Node {} has no id column; skipping graph features for edge {}.",
                node.alias,
                edge.alias,
            )
            return

        causal = f" where {edge.timestamp} <= aod.as_of_date " if edge.timestamp else ""
        weight_expr = edge.weight if edge.weight else "null"
        union = (
            f"select {edge.source} as node_id, 'out' as direction, "
            f"{weight_expr} as weight from {edge.table}{causal} "
            "union all "
            f"select {edge.target} as node_id, 'in' as direction, "
            f"{weight_expr} as weight from {edge.table}{causal}"
        )

        def feature_name(metric: str) -> str:
            return f'"{metric}({node.alias}.{edge.alias})"'

        columns = [
            (feature_name("OUT_DEGREE"), "count(*) filter (where direction = 'out')"),
            (feature_name("IN_DEGREE"), "count(*) filter (where direction = 'in')"),
            (feature_name("DEGREE"), "count(*)"),
        ]
        if edge.weight:
            columns.extend(
                [
                    (
                        feature_name("WEIGHTED_OUT_DEGREE"),
                        "coalesce(sum(weight) filter (where direction = 'out'), 0)",
                    ),
                    (
                        feature_name("WEIGHTED_IN_DEGREE"),
                        "coalesce(sum(weight) filter (where direction = 'in'), 0)",
                    ),
                ]
            )

        select_cols = ",\n        ".join(f"{expr} as {name}" for name, expr in columns)
        cte_name = f"{edge.alias}_graph_for_{node.alias}"
        cte_query = f"""
        -- graph (degree) features for {node.alias} over edge {edge.alias}
        {cte_name} as (
        select node_id,
        {select_cols}
        from ( {union} ) as incident_{edge.alias}
        group by node_id
        )
        """
        join = f" {cte_name} on {cte_name}.node_id = {node.table}.{node.id.name} "
        self._joins[node.alias].append(join)
        self._ctes.append(cte_query)

        # Register degree features so they flow through synth/transform and are
        # available for downstream transformation and parent aggregation. They
        # are synth columns, so Fix A references them by name in the transform.
        self._features[node.alias].update(
            Feature(name=name, type="numeric", definition=name, entity=node)
            for name, _ in columns
        )

    def _build_aggregations(
        self, target: Entity, source: Entity, relationship: Relationship
    ) -> None:
        logger.debug("Processing backward relationship {}", relationship)
        aggregations: List[Feature] = []

        for feature in self._features[source.alias]:
            for aggregator in self.aggregations.values():
                new_feature = aggregator(
                    target, source, feature, relationship=relationship
                )
                if new_feature:
                    aggregations.append(new_feature)

                for interval in self.intervals:
                    if feature.entity is None or feature.entity.temporal_ix is None:
                        logger.warning(
                            "Entity {} lacks temporal index; skipping interval-based aggregation",
                            feature.entity,
                        )
                        break
                    interval_feature = aggregator(
                        target,
                        source,
                        feature,
                        interval=interval,
                        relationship=relationship,
                    )
                    if interval_feature:
                        aggregations.append(interval_feature)

        aggregation_set = set(aggregations)
        sorted_aggs = self._sort_features(aggregation_set)
        self._debug(
            "aggregations",
            target=target.alias,
            source=source.alias,
            count=len(sorted_aggs),
        )

        self._features[source.alias].update(aggregation_set)
        self._features[target.alias].update(aggregation_set)

        self._build_aggregations_cte(target, source, relationship, sorted_aggs)

    def _build_direct(
        self, target: Entity, source: Entity, relationship: Relationship
    ) -> None:
        logger.debug("Processing forward relationship {}", relationship)
        directs = self._sort_features(self._features[source.alias])
        self._debug(
            "direct_features",
            target=target.alias,
            source=source.alias,
            count=len(directs),
        )

        self._features[target.alias].update(directs)
        if getattr(relationship, "temporal_mode", None) == "as_of":
            self._build_direct_asof(target, source, relationship, directs)
        else:
            self._build_direct_cte(target, source, relationship, directs)

    def _build_transformations(self, target: Entity) -> None:
        self._build_synth_cte(target)

        transformed: List[Feature | Iterable[Feature]] = []
        for feature in self._features[target.alias]:
            if feature.type != "index":
                for transformer in self.transformations.values():
                    new_feature = transformer(target, feature)
                    if new_feature:
                        transformed.append(new_feature)
            else:
                transformed.append(feature)

        flattened: Set[Feature] = set()
        for candidate in transformed:
            if isinstance(candidate, Feature):
                flattened.add(candidate)
            elif isinstance(candidate, (list, tuple, set)):
                for item in candidate:
                    if isinstance(item, Feature):
                        flattened.add(item)

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

        rendered_features = [
            feature.query for feature in features if feature.type not in ["key"]
        ]
        where_clause = (
            f"where aod.as_of_date >= {source.temporal_ix.name}"
            if source.temporal_ix
            else ""
        )

        cte_query = f"""
        -- Aggregate for {target.alias}
        {cte_name} as (
        select
        {source.alias}_transform.{relationship.parent_key},
        {",".join(rendered_features)}
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
            logger.debug(
                "Skipping direct features for {} because it lacks an id column.",
                source.alias,
            )
            return

        cte_name = f"{source.alias}_direct_transfers_for_{target.alias}"

        feature_names = [
            feature.name for feature in features if feature.type not in ["index", "key"]
        ]
        cte_query = f"""
        -- direct features for {target.alias}
        {cte_name} as (
        select
        {source.id.name},
        {",".join(feature_names)}
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
        source_temporal = relationship.temporal_child_field or (
            source.temporal_ix.name if source.temporal_ix else None
        )
        if not target_temporal or not source_temporal:
            logger.warning(
                "Temporal join requested between {} and {} but temporal indexes are missing; falling back to static join.",
                source.alias,
                target.alias,
            )
            self._build_direct_cte(target, source, relationship, features)
            return

        # feature.name is already a quoted identifier for aggregate/transform
        # features (e.g. "ABS(care_plans.risk_score)"); wrapping it in another
        # pair of quotes yields an empty delimited identifier. It is also the
        # column name projected by <source>_transform, so reference it as-is.
        projected = [
            f"{source.alias}_transform.{feature.name} as {feature.name}"
            for feature in features
            if feature.type not in {"index", "key"}
        ]
        if not projected:
            return

        projected_sql = ",\n        ".join(projected)

        where_clauses = [
            f"{source.alias}_transform.{relationship.parent_key} = {target.table}.{relationship.child_key}",
            f"{source.alias}_transform.{source_temporal} <= {target.table}.{target_temporal}",
        ]
        if relationship.temporal_grace:
            # Lower-bound the lookback to the grace window. Written as
            # ``source >= target - interval`` (equivalent to
            # ``target - source <= grace``) so it is valid for both date and
            # timestamp temporals: ``date - date`` yields an integer, which
            # cannot be compared against an interval, but ``date - interval``
            # yields a timestamp that compares cleanly.
            where_clauses.append(
                f"{source.alias}_transform.{source_temporal} >= "
                f"{target.table}.{target_temporal} - interval '{relationship.temporal_grace}'"
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

        id_columns = [
            f"{target.table}.{name}" for name in self._identifier_columns(target)
        ]
        feature_names = [
            feature.name
            for feature in self._sort_features(self._features[target.alias])
            if feature.type not in ["index", "key"]
        ]

        # Record what synth projects so the transform CTE can reference these
        # columns by name instead of re-rendering their (base-table) definitions.
        self._synth_columns[target.alias] = set(feature_names)

        cte_query = f"""
        -- sythetize aggregations and direct features for {target.alias}
        {cte_table} as (
        select
        {", ".join(id_columns + feature_names)}
        from {target.table}
        {" left join " if self._joins[target.alias] else ""}
        {" left join ".join(self._joins[target.alias])}
        )
        """

        self._ctes.append(cte_query)

    def _build_transform_cte(self, target: Entity, features: Iterable[Feature]) -> None:
        cte_table = f"{target.alias}_transform"

        id_columns = self._identifier_columns(target)
        synth_columns = self._synth_columns.get(target.alias, set())
        rendered_features = []
        for feature in features:
            if feature.type in ["index", "key"]:
                continue
            if feature.name in synth_columns:
                # Already a column in <target>_synth (an aggregate carried up
                # from a child, an as-of/direct pull, or a base variable). Its
                # own definition references base-table columns absent from synth,
                # so reference it by name. feature.name is already quoted when
                # it needs to be.
                rendered_features.append(f"{feature.name} as {feature.name}")
            else:
                # A genuine transformer output; its definition references synth
                # columns by name and is valid against the synth CTE.
                rendered_features.append(feature.query)

        cte_query = f"""
        -- transform {target.alias}
        {cte_table} as (
        select
        {", ".join(id_columns + rendered_features)}
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
