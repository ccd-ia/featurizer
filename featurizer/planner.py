# coding: utf-8

"""Feature planning orchestration.

The planner traverses the entity graph, synthesizes features, and collects the
CTE definitions/joins that the SQL renderer expects.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Set

from icecream import ic
from loguru import logger

from .primitives import Entity, ERGraph, Feature, Relationship, pg_identifier


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
        self._build_peer_group_features(target_entity)
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
        """Attach the requested graph feature families for every edge on this node."""
        for edge in self.graph.get_edges_for_node(node):
            self._build_graph_cte(node, edge)

    def _build_graph_cte(self, node: Entity, edge) -> None:
        """Emit one CTE per requested graph family for ``node`` over ``edge``.

        Families are requested via ``edge: {features: [...]}`` (default
        ``[degree]``). When the edge carries a ``timestamp`` every family is
        bounded by ``<= aod.as_of_date`` so the graph is measured as-of each
        cutoff (the same causal guarantee the aggregation CTEs use); without
        one the graph is treated as static and leakage is the caller's
        responsibility.

        The recursive families (k_hop_2, clustering, common_neighbours,
        jaccard, adamic_adar) share an undirected, deduplicated neighbour CTE;
        reciprocity reads the raw directed edge table; degree keeps its
        original union-by-direction shape.
        """
        if node.id is None:
            logger.warning(
                "Node {} has no id column; skipping graph features for edge {}.",
                node.alias,
                edge.alias,
            )
            return

        node_id_col = node.id.name
        families = list(edge.features)
        registered: list[str] = []

        def attach(cte_name: str, cte_query: str, names: list[str]) -> None:
            join = f" {cte_name} on {cte_name}.node_id = {node.table}.{node_id_col} "
            self._joins[node.alias].append(join)
            self._ctes.append(cte_query)
            registered.extend(names)

        if "degree" in families:
            self._graph_degree_cte(node, edge, attach)
        if "reciprocity" in families:
            self._graph_reciprocity_cte(node, edge, attach)

        neighbour_families = [
            f
            for f in (
                "k_hop_2",
                "clustering",
                "common_neighbours",
                "jaccard",
                "adamic_adar",
            )
            if f in families
        ]
        if neighbour_families:
            nbrs = self._graph_neighbours_cte(node, edge)
            if "k_hop_2" in neighbour_families:
                self._graph_k_hop_cte(node, edge, nbrs, attach)
            if "clustering" in neighbour_families:
                self._graph_clustering_cte(node, edge, nbrs, attach)
            linkpred = [
                f
                for f in ("common_neighbours", "jaccard", "adamic_adar")
                if f in neighbour_families
            ]
            if linkpred:
                self._graph_linkpred_cte(node, edge, nbrs, linkpred, attach)

        # Register graph features so they flow through synth/transform and are
        # available for downstream transformation and parent aggregation. They
        # are synth columns, so Fix A references them by name in the transform.
        self._features[node.alias].update(
            Feature(name=name, type="numeric", definition=name, entity=node)
            for name in registered
        )

    @staticmethod
    def _graph_feature_name(metric: str, node: Entity, edge) -> str:
        return pg_identifier(f"{metric}({node.alias}.{edge.alias})")

    @staticmethod
    def _graph_causal(edge, *, prefix: str, alias: str = "") -> str:
        """Causal bound on the edge timestamp; empty for static graphs."""
        if not edge.timestamp:
            return ""
        col = f"{alias}.{edge.timestamp}" if alias else edge.timestamp
        return f" {prefix} {col} <= aod.as_of_date"

    def _graph_degree_cte(self, node: Entity, edge, attach) -> None:
        causal = self._graph_causal(edge, prefix="where")
        weight_expr = edge.weight if edge.weight else "null"
        union = (
            f"select {edge.source} as node_id, 'out' as direction, "
            f"{weight_expr} as weight from {edge.table}{causal} "
            "union all "
            f"select {edge.target} as node_id, 'in' as direction, "
            f"{weight_expr} as weight from {edge.table}{causal}"
        )

        columns = [
            (
                self._graph_feature_name("OUT_DEGREE", node, edge),
                "count(*) filter (where direction = 'out')",
            ),
            (
                self._graph_feature_name("IN_DEGREE", node, edge),
                "count(*) filter (where direction = 'in')",
            ),
            (self._graph_feature_name("DEGREE", node, edge), "count(*)"),
        ]
        if edge.weight:
            columns.extend(
                [
                    (
                        self._graph_feature_name("WEIGHTED_OUT_DEGREE", node, edge),
                        "coalesce(sum(weight) filter (where direction = 'out'), 0)",
                    ),
                    (
                        self._graph_feature_name("WEIGHTED_IN_DEGREE", node, edge),
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
        attach(cte_name, cte_query, [name for name, _ in columns])

    def _graph_reciprocity_cte(self, node: Entity, edge, attach) -> None:
        """Fraction of a node's outgoing edges that are reciprocated.

        Reads the *directed* edge table: an edge s->t is reciprocated when
        t->s also exists (within the causal bound). On an edge table that
        stores unordered pairs once (deduplicated undirected graphs) this is
        0 by construction.
        """
        name = self._graph_feature_name("RECIPROCITY", node, edge)
        causal_outer = self._graph_causal(edge, prefix="where", alias="e")
        causal_inner = self._graph_causal(edge, prefix="and", alias="r")
        cte_name = f"{edge.alias}_recip_for_{node.alias}"
        cte_query = f"""
        -- graph (reciprocity) for {node.alias} over edge {edge.alias}
        {cte_name} as (
        select e.{edge.source} as node_id,
        (count(*) filter (where exists (
            select 1 from {edge.table} r
            where r.{edge.source} = e.{edge.target}
              and r.{edge.target} = e.{edge.source}{causal_inner}
        )))::float / count(*) as {name}
        from {edge.table} e{causal_outer}
        group by e.{edge.source}
        )
        """
        attach(cte_name, cte_query, [name])

    def _graph_neighbours_cte(self, node: Entity, edge) -> str:
        """Emit (once) the shared undirected, deduplicated neighbour CTE."""
        cte_name = f"{edge.alias}_nbrs_for_{node.alias}"
        causal = self._graph_causal(edge, prefix="where")
        cte_query = f"""
        -- shared undirected neighbour list for {node.alias} over edge {edge.alias}
        {cte_name} as (
        select distinct node_id, nbr from (
            select {edge.source} as node_id, {edge.target} as nbr
            from {edge.table}{causal}
            union all
            select {edge.target} as node_id, {edge.source} as nbr
            from {edge.table}{causal}
        ) u where node_id is not null and nbr is not null
        )
        """
        self._ctes.append(cte_query)
        return cte_name

    def _graph_k_hop_cte(self, node: Entity, edge, nbrs: str, attach) -> None:
        """Count of distinct nodes at exactly 2 hops (not ego, not a neighbour)."""
        name = self._graph_feature_name("K_HOP_2_COUNT", node, edge)
        cte_name = f"{edge.alias}_k2_for_{node.alias}"
        cte_query = f"""
        -- graph (2-hop neighbourhood size) for {node.alias} over edge {edge.alias}
        {cte_name} as (
        select n1.node_id,
        count(distinct n2.nbr) as {name}
        from {nbrs} n1
        join {nbrs} n2 on n2.node_id = n1.nbr
        where n2.nbr <> n1.node_id
          and not exists (
            select 1 from {nbrs} d
            where d.node_id = n1.node_id and d.nbr = n2.nbr
          )
        group by n1.node_id
        )
        """
        attach(cte_name, cte_query, [name])

    def _graph_clustering_cte(self, node: Entity, edge, nbrs: str, attach) -> None:
        """Local clustering coefficient: connected neighbour pairs / possible pairs."""
        name = self._graph_feature_name("CLUSTERING_COEFF", node, edge)
        cte_name = f"{edge.alias}_clust_for_{node.alias}"
        cte_query = f"""
        -- graph (local clustering coefficient) for {node.alias} over edge {edge.alias}
        {cte_name} as (
        select deg.node_id,
        case when deg.degree < 2 then null
             else coalesce(tri.closed, 0)::float
                  / (deg.degree * (deg.degree - 1) / 2.0)
        end as {name}
        from (
            select node_id, count(*) as degree from {nbrs} group by node_id
        ) deg
        left join (
            select n1.node_id, count(*) as closed
            from {nbrs} n1
            join {nbrs} n2 on n2.node_id = n1.node_id and n2.nbr > n1.nbr
            join {nbrs} e on e.node_id = n1.nbr and e.nbr = n2.nbr
            group by n1.node_id
        ) tri on tri.node_id = deg.node_id
        )
        """
        attach(cte_name, cte_query, [name])

    def _graph_linkpred_cte(
        self, node: Entity, edge, nbrs: str, families: list[str], attach
    ) -> None:
        """Ego-level link-prediction scores, averaged over the ego's neighbours.

        For each neighbour v of ego e: common neighbours |N(e) ∩ N(v)|,
        Jaccard |N(e) ∩ N(v)| / |N(e) ∪ N(v)|, and Adamic-Adar
        Σ_w 1/ln(deg(w)) over the common neighbours w. Pairs with no common
        neighbours contribute 0 to the means.
        """
        columns = []
        if "common_neighbours" in families:
            columns.append(
                (
                    self._graph_feature_name("COMMON_NEIGHBOURS_MEAN", node, edge),
                    "avg(coalesce(cn.cnt, 0))",
                )
            )
        if "jaccard" in families:
            columns.append(
                (
                    self._graph_feature_name("JACCARD_MEAN", node, edge),
                    "avg(coalesce(cn.cnt, 0)::float / "
                    "nullif(de.degree + dv.degree - coalesce(cn.cnt, 0), 0))",
                )
            )
        if "adamic_adar" in families:
            columns.append(
                (
                    self._graph_feature_name("ADAMIC_ADAR_MEAN", node, edge),
                    "avg(coalesce(cn.aa, 0))",
                )
            )
        select_cols = ",\n        ".join(f"{expr} as {name}" for name, expr in columns)

        cte_name = f"{edge.alias}_linkpred_for_{node.alias}"
        cte_query = f"""
        -- graph (link-prediction means) for {node.alias} over edge {edge.alias}
        {cte_name} as (
        with degrees as (
            select node_id, count(*) as degree from {nbrs} group by node_id
        ),
        pair_cn as (
            select n1.node_id as ego, n1.nbr as v,
                   count(n2.nbr) as cnt,
                   sum(1.0 / nullif(ln(dw.degree), 0)) as aa
            from {nbrs} n1
            join {nbrs} n2 on n2.node_id = n1.node_id
            join {nbrs} nv on nv.node_id = n1.nbr and nv.nbr = n2.nbr
            join degrees dw on dw.node_id = n2.nbr
            where n2.nbr <> n1.nbr
            group by n1.node_id, n1.nbr
        )
        select p.node_id,
        {select_cols}
        from {nbrs} p
        join degrees de on de.node_id = p.node_id
        join degrees dv on dv.node_id = p.nbr
        left join pair_cn cn on cn.ego = p.node_id and cn.v = p.nbr
        group by p.node_id
        )
        """
        attach(cte_name, cte_query, [name for name, _ in columns])

    # ------------------------------------------------------------------ #
    # Peer-group features (M1d Phase 8)
    # ------------------------------------------------------------------ #

    def _build_peer_group_features(self, entity: Entity) -> None:
        """Attach peer-group features for every ``peer_groups`` spec on ``entity``.

        Peers of a row are the other rows of the *same* entity sharing the
        categorical ``by`` column. Every feature is leave-one-out (the ego is
        removed from its own peer aggregate) and bounded ``<= aod.as_of_date``
        on both peer membership (the entity's own ``temporal_ix``, when present)
        and the peers' child event stream — the highest-leakage surface in the
        program, so the causal cut is explicit at every read.
        """
        specs = getattr(entity, "peer_groups", None)
        if not specs:
            return
        if entity.id is None:
            logger.warning(
                "Entity {} has no id column; skipping peer-group features.",
                entity.alias,
            )
            return

        # One shared per-peer event-count CTE per backward child stream, emitted
        # once and reused across every peer_groups spec on this entity.
        child_count_ctes = self._peer_child_count_ctes(entity)
        for spec in specs:
            self._build_peer_group_cte(entity, spec, child_count_ctes)

    @staticmethod
    def _peer_feature_name(
        metric: str,
        entity: Entity,
        by: str,
        measure: str | None = None,
        *,
        child: str | None = None,
    ) -> str:
        if child is not None:
            inner = f"{entity.alias}.{child} by {by}"
        elif measure is not None:
            inner = f"{entity.alias}.{measure} by {by}"
        else:
            inner = f"{entity.alias} by {by}"
        return pg_identifier(f"{metric}({inner})")

    @staticmethod
    def _numeric_variable_names(entity: Entity) -> List[str]:
        """Default ``measures``: the entity's numeric variables (sorted)."""
        return sorted(
            feature.name for feature in entity.features if feature.type == "numeric"
        )

    def _peer_child_count_ctes(self, entity: Entity) -> List[tuple[Relationship, str]]:
        """Emit (once) a per-peer event-count CTE for each backward child stream.

        Each CTE maps a peer's id to its count of child rows knowable as-of the
        cutoff; the main peer CTE then averages these over the peer set.
        """
        results: List[tuple[Relationship, str]] = []
        relationships = sorted(
            self.graph.get_backward_relationships(entity),
            key=lambda rel: rel.child.alias,
        )
        for rel in relationships:
            child = rel.child
            child_temporal = child.temporal_ix.name if child.temporal_ix else None
            causal = (
                f"where c.{child_temporal} <= aod.as_of_date" if child_temporal else ""
            )
            cte_name = f"peer_evt_{child.alias}_for_{entity.alias}"
            cte_query = f"""
        -- per-peer event counts ({child.alias}) for {entity.alias} peer groups
        {cte_name} as (
        select c.{rel.child_key} as pid, count(*) as cnt
        from {child.table} c
        {causal}
        group by c.{rel.child_key}
        )
        """
            self._ctes.append(cte_query)
            results.append((rel, cte_name))
        return results

    def _build_peer_group_cte(
        self,
        entity: Entity,
        spec,
        child_count_ctes: List[tuple[Relationship, str]],
    ) -> None:
        assert entity.id is not None  # guarded by _build_peer_group_features
        by = spec.by
        id_col = entity.id.name
        table = entity.table
        temporal = entity.temporal_ix.name if entity.temporal_ix else None

        # 1 when the ego itself is a peer (a member knowable as-of the cutoff),
        # so leave-one-out subtracts the ego only when it belongs to the set.
        in_grp = (
            f"(case when e.{temporal} <= aod.as_of_date then 1 else 0 end)"
            if temporal
            else "1"
        )
        membership = f"where e2.{temporal} <= aod.as_of_date" if temporal else ""
        n_excl = f"(g.n - {in_grp})"

        grp_cols: List[str] = [f"e2.{by} as grp", "count(*) as n"]
        grp_joins: List[str] = []
        ego_joins: List[str] = []
        select_cols: List[tuple[str, str]] = []

        # Peer-set size (leave-one-out), always emitted.
        select_cols.append(
            (self._peer_feature_name("PEER_GROUP_SIZE", entity, by), n_excl)
        )

        # Per-measure attribute statistics (mean / delta / z-score / percentile).
        measures = spec.measures
        if measures is None:
            measures = self._numeric_variable_names(entity)
        for measure in measures:
            grp_cols.append(f"sum(e2.{measure}) as sum_{measure}")
            grp_cols.append(f"sum(e2.{measure} * e2.{measure}) as ss_{measure}")
            sum_excl = f"(g.sum_{measure} - {in_grp} * e.{measure})"
            ss_excl = f"(g.ss_{measure} - {in_grp} * e.{measure} * e.{measure})"
            mean_excl = f"({sum_excl} / nullif({n_excl}, 0))"
            var_excl = (
                f"(({ss_excl} - {sum_excl} * {sum_excl} / nullif({n_excl}, 0)) "
                f"/ nullif({n_excl} - 1, 0))"
            )
            std_excl = f"sqrt(greatest({var_excl}, 0))"
            peer_causal = f" and p.{temporal} <= aod.as_of_date" if temporal else ""
            pctile = (
                f"((select count(*) from {table} p where p.{by} = e.{by} "
                f"and p.{id_col} <> e.{id_col} and p.{measure} < e.{measure}"
                f"{peer_causal})::float / nullif({n_excl}, 0))"
            )
            select_cols.extend(
                [
                    (
                        self._peer_feature_name("PEER_MEAN", entity, by, measure),
                        mean_excl,
                    ),
                    (
                        self._peer_feature_name(
                            "EGO_MINUS_PEER_MEAN", entity, by, measure
                        ),
                        f"(e.{measure} - {mean_excl})",
                    ),
                    (
                        self._peer_feature_name("PEER_ZSCORE", entity, by, measure),
                        f"((e.{measure} - {mean_excl}) / nullif({std_excl}, 0))",
                    ),
                    (
                        self._peer_feature_name("PEER_PCTILE", entity, by, measure),
                        pctile,
                    ),
                ]
            )

        # Cross-stream peer event rate: mean per-peer child-event count.
        for rel, cte_name in child_count_ctes:
            child = rel.child.alias
            grp_cols.append(f"sum(coalesce(pc_{child}.cnt, 0)) as sum_evt_{child}")
            grp_joins.append(
                f"left join {cte_name} pc_{child} on pc_{child}.pid = e2.{id_col}"
            )
            ego_joins.append(
                f"left join {cte_name} ec_{child} on ec_{child}.pid = e.{id_col}"
            )
            rate = (
                f"((g.sum_evt_{child} - {in_grp} * coalesce(ec_{child}.cnt, 0))::float "
                f"/ nullif({n_excl}, 0))"
            )
            select_cols.append(
                (
                    self._peer_feature_name("PEER_EVENT_RATE", entity, by, child=child),
                    rate,
                )
            )

        grp_subquery = (
            f"select {', '.join(grp_cols)} "
            f"from {table} e2 {' '.join(grp_joins)} {membership} "
            f"group by e2.{by}"
        )
        rendered = ",\n        ".join(f"{expr} as {name}" for name, expr in select_cols)
        cte_name = f"peer_{by}_for_{entity.alias}"
        cte_query = f"""
        -- peer-group features for {entity.alias} grouped by {by}
        {cte_name} as (
        select e.{id_col} as node_id,
        {rendered}
        from {table} e
        join ( {grp_subquery} ) g on g.grp = e.{by}
        {' '.join(ego_joins)}
        )
        """
        join = f" {cte_name} on {cte_name}.node_id = {table}.{id_col} "
        self._joins[entity.alias].append(join)
        self._ctes.append(cte_query)
        self._features[entity.alias].update(
            Feature(name=name, type="numeric", definition=name, entity=entity)
            for name, _ in select_cols
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
