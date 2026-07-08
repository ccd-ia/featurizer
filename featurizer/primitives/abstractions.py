# coding: utf-8

from __future__ import annotations

import hashlib
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .preagg import PreAggSpec

# Canonical feature-type vocabulary. The user-declarable variable types
# (numeric, categorical, text, boolean, date, timestamp, vector) are kept in
# sync with ``ConfigValidator.VALID_VARIABLE_TYPES``; the remaining members
# (index, spatial_ix, temporal_ix, key) are internal index roles assigned by
# ``Entity`` construction, not declared in ``variables:``. ``vector`` carries
# pgvector / embedding columns materialized by the bridge layer.
FeatureType = Enum(
    "FeatureType",
    "index spatial_ix temporal_ix date timestamp numeric categorical text boolean vector key",
)

#: Graph feature families an edge-table entity may request via
#: ``edge: {features: [...]}``. ``degree`` is the backward-compatible default.
GRAPH_FEATURE_FAMILIES = (
    "degree",
    "reciprocity",
    "k_hop_2",
    "clustering",
    "common_neighbours",
    "jaccard",
    "adamic_adar",
)


def _truncate_identifier(raw: str) -> str:
    """Cap a name at PostgreSQL's 63-byte identifier limit, deterministically.

    PostgreSQL truncates identifiers to 63 bytes (NAMEDATALEN - 1); two long
    names sharing a 63-byte prefix would silently collide into one ambiguous
    column, so long names are capped with a stable hash suffix (bug #8). The
    suffix is the first 8 hex chars (32 bits) of the MD5 of the *full* name,
    so the mapping is identical across processes and runs. With feature counts
    in the thousands, the birthday-collision probability over a shared 54-char
    prefix is negligible (~n^2 / 2^33). Returns the bare identifier (no quotes).
    """
    raw = raw.replace('"', "")
    if len(raw.encode()) > 63:
        digest = hashlib.md5(raw.encode()).hexdigest()[:8]
        raw = f"{raw[:54]}~{digest}"
    return raw


def pg_identifier(raw: str) -> str:
    """Quote a generated feature name as a PostgreSQL identifier.

    Long names are capped via :func:`_truncate_identifier` before quoting so
    they survive PostgreSQL's 63-byte cap without silently colliding (bug #8).
    """
    return f'"{_truncate_identifier(raw)}"'


class ERGraph:
    def __init__(
        self,
        entities: List[Dict[str, Any]],
        relationships: Optional[List[Dict[str, Any]]],
        spatial_relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.entities: Dict[str, Entity] = {e["alias"]: Entity(**e) for e in entities}

        self.relationships: List[Relationship]
        if relationships:
            self.relationships = [
                Relationship(
                    parent=self.entities[r["parent"]["entity"]],
                    child=self.entities[r["child"]["entity"]],
                    parent_key=r["parent"]["key"],
                    child_key=r["child"]["key"],
                    temporal_mode=(r.get("temporal") or {}).get("mode"),
                    temporal_grace=(r.get("temporal") or {}).get("grace"),
                    temporal_child_field=(r.get("temporal") or {}).get(
                        "child_timestamp"
                    ),
                    name=r.get("name"),
                )
                for r in relationships
            ]
        else:
            self.relationships = []

        for r in self.relationships:
            self.entities[r.child.alias].add_key(Key(name=r.child_key, entity=r.child))
            # The parent side must carry its join column too: the direct and
            # as-of builders read <parent>_transform.<parent_key>, which is only
            # projected when the column is an index/key. When parent_key is the
            # entity id (the common case) the by-name dedupe in
            # _identifier_columns keeps projections unchanged.
            self.entities[r.parent.alias].add_key(
                Key(name=r.parent_key, entity=r.parent)
            )

        # Edge-table entities contribute graph features to their node entity.
        self.edges: List[EdgeSpec] = [
            e.edge for e in self.entities.values() if e.edge is not None
        ]

        # Spatial second-table relationships (co-location / nearest / KDE).
        self.spatial_relationships: List[SpatialRelationshipSpec] = [
            SpatialRelationshipSpec(
                name=s["name"],
                left=s["left"],
                right=s["right"],
                within_m=s["within_m"],
                bandwidth_m=s.get("bandwidth_m"),
                features=s.get("features"),
            )
            for s in (spatial_relationships or [])
        ]

    def get_edges_for_node(self, entity: Entity) -> List["EdgeSpec"]:
        """Edge specs whose node alias matches ``entity``."""
        return [edge for edge in self.edges if edge.node == entity.alias]

    def get_backward_entities(self, entity: Entity) -> Set[Entity]:
        return {r.child for r in self.relationships if r.parent == entity}

    def get_forward_entities(self, entity: Entity) -> Set[Entity]:
        return {r.parent for r in self.relationships if r.child == entity}

    def get_backward_relationships(self, entity: Entity) -> Set[Relationship]:
        return {r for r in self.relationships if r.parent == entity}

    def get_forward_relationships(self, entity: Entity) -> Set[Relationship]:
        return {r for r in self.relationships if r.child == entity}


class Entity:
    def __init__(
        self,
        alias: str,
        table: str,
        id: Optional[str] = None,
        spatial_ix: Any = None,
        temporal_ix: Optional[str] = None,
        variables: Optional[Dict[str, Dict[str, Any]]] = None,
        edge: Optional[Dict[str, Any]] = None,
        peer_groups: Optional[List[Dict[str, Any]]] = None,
        peer_group: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.alias: str = alias
        self.table: str = table

        self.id: Optional[Id] = Id(name=id, entity=self) if id else None
        # spatial_ix is either a single column (geometry/point, backward-compatible)
        # or a {lat, lon} / {geom} dict parsed into a SpatialIx (plain-SQL path).
        self.spatial_ix = self._build_spatial_ix(spatial_ix)  # "event location"
        self.temporal_ix: Optional[Id] = (
            Id(name=temporal_ix, entity=self) if temporal_ix else None
        )  # Temporal index a.k.a "event date"

        self.keys: List[Key] = []  # Foreign keys to another dataset

        self.features: List[Feature] = []

        if variables is not None:
            self.features = [
                Variable(
                    name=var,
                    type=description["type"],
                    entity=self,
                    predicates=description.get("predicates"),
                    role=description.get("role"),
                    vocabulary=description.get("vocabulary"),
                )
                for var, description in variables.items()
            ]

        self.features = (
            self.features
            + ([self.id] if self.id else [])
            + ([self.temporal_ix] if self.temporal_ix else [])
            + self._spatial_features()
        )

        # An edge-table entity: rows are graph edges between nodes. The planner
        # attaches graph features (degree, ...) to the referenced node entity
        # rather than treating this as a normal aggregation child.
        self.edge: Optional[EdgeSpec] = (
            EdgeSpec(
                entity=self,
                node=edge["node"],
                source=edge["source"],
                target=edge["target"],
                weight=edge.get("weight"),
                timestamp=edge.get("timestamp"),
                features=edge.get("features"),
            )
            if edge is not None
            else None
        )

        # Peer-group specs: compare each row to its peers (rows of this same
        # entity sharing a categorical column). ``peer_group`` (singular dict)
        # is sugar for a one-element ``peer_groups`` list.
        peer_specs: List[Dict[str, Any]] = []
        if peer_group is not None:
            peer_specs.append(peer_group)
        if peer_groups is not None:
            peer_specs.extend(peer_groups)
        self.peer_groups: List[PeerGroupSpec] = [
            PeerGroupSpec(entity=self, by=spec["by"], measures=spec.get("measures"))
            for spec in peer_specs
        ]

    def _build_spatial_ix(self, spatial_ix: Any):
        """Parse spatial_ix: a column name (Id) or a {lat,lon}/{geom} SpatialIx."""
        if spatial_ix is None:
            return None
        if isinstance(spatial_ix, dict):
            return SpatialIx(
                entity=self,
                lat=spatial_ix.get("lat"),
                lon=spatial_ix.get("lon"),
                geom=spatial_ix.get("geom"),
                srid=spatial_ix.get("srid", 4326),
            )
        return Id(name=spatial_ix, entity=self)

    def _spatial_features(self) -> List[Id]:
        """Spatial component columns to carry through the CTEs (as index features)."""
        if isinstance(self.spatial_ix, Id):
            return [self.spatial_ix]
        if isinstance(self.spatial_ix, SpatialIx):
            return list(self.spatial_ix.columns)
        return []

    def __repr__(self) -> str:
        return f"Entity({self.alias})"

    def info(self) -> str:
        feature_list = ", ".join(
            f.name for f in self.features if isinstance(f, Variable)
        )
        return f"""

        {self.alias.capitalize()}(table = {self.table})

            Variables:
               {feature_list}

        """

    @property
    def indexes(self) -> List[Id]:
        result: List[Optional[Id]] = [self.id]
        result.extend(self._spatial_features())
        result.append(self.temporal_ix)
        return [ix for ix in result if ix is not None]

    def add_key(self, key: Key) -> None:
        if key not in self.keys:
            self.keys.append(key)

    def add_features(self, features: List[Feature]) -> None:
        for feature in features:
            if feature not in self.features:
                self.features.append(feature)


class Relationship:
    def __init__(
        self,
        parent: Entity,
        child: Entity,
        parent_key: str,
        child_key: str,
        temporal_mode: Optional[str] = None,
        temporal_grace: Optional[str] = None,
        temporal_child_field: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.parent: Entity = parent
        self.parent_key: str = parent_key
        self.child: Entity = child
        self.child_key: str = child_key
        self.temporal_mode: Optional[str] = temporal_mode
        self.temporal_grace: Optional[str] = temporal_grace
        self.temporal_child_field: Optional[str] = temporal_child_field
        # Optional relationship name (config ``relationships[].name``). Gives
        # parallel relationships between one entity pair distinct identities:
        # it replaces the child alias in aggregation feature/CTE names and
        # qualifies direct/as-of transferred columns. When unset (every config
        # that was unambiguous before v0.5.0), naming falls back to the entity
        # aliases so existing feature names stay byte-identical.
        self.name: Optional[str] = name

    @property
    def naming_alias(self) -> str:
        """Alias used in aggregation feature names and the aggs CTE name."""
        return self.name or self.child.alias

    @property
    def parent_naming_alias(self) -> str:
        """Alias used in the direct-transfer / as-of CTE names."""
        return self.name or self.parent.alias

    def __repr__(self) -> str:
        label = f" [{self.name}]" if self.name else ""
        return (
            f"{self.parent}.{self.parent_key} -> "
            f"{self.child}.{self.child_key}{label}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relationship):
            return False
        return (
            self.parent == other.parent
            and self.parent_key == other.parent_key
            and self.child == other.child
            and self.child_key == other.child_key
            and self.temporal_mode == other.temporal_mode
            and self.temporal_grace == other.temporal_grace
            and self.temporal_child_field == other.temporal_child_field
            and self.name == other.name
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.parent.alias,
                self.parent_key,
                self.child.alias,
                self.child_key,
                self.temporal_mode,
                self.temporal_grace,
                self.temporal_child_field,
                self.name,
            )
        )

    def __contains__(self, entity: Entity) -> bool:
        if entity in [self.parent, self.child]:
            return True
        return False

    def is_backward(self, e1: Entity, e2: Entity) -> bool:
        return e1 == self.parent and e2 == self.child

    def is_forward(self, e1: Entity, e2: Entity) -> bool:
        return e1 == self.child and e2 == self.parent


class Feature:
    """Base class for features"""

    def __init__(
        self,
        name: str,
        type: str,
        definition: Optional[str] = None,
        entity: Optional[Entity] = None,
        parents: Optional[List[Feature]] = None,
        intervals: Optional[List[str]] = None,
        specials: Optional[List[Any]] = None,
        sort: Optional[str] = None,
        description: str = "a feature",
        stack_depth: int = 0,
        predicates: Optional[Dict[str, str]] = None,
        label: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self.type: str = type
        self.definition: Optional[str] = definition
        # Full, untruncated, human-readable intended name. ``name`` may be a
        # quoted + 63-byte-capped ``pg_identifier`` (the hash suffix erases the
        # tail of long generated names); ``label`` preserves what truncation
        # throws away so the feature manifest can map column -> intended name.
        # Defaults to ``name`` for plain columns (name == intended).
        self.label: str = label if label is not None else name
        # For direct-transfer features renamed by a *named* relationship
        # (``"<rel>.<column>"``): the original source-side column name the
        # transfer CTE must read. None for every other feature.
        self.direct_source: Optional[str] = None
        # For subquery aggregators that opted into the set-based rewrite: the
        # pre-aggregation spec routing this feature into a companion CTE instead
        # of a correlated subquery (ADR-0010). None for every correlated /
        # plain-GROUP-BY feature. Does not participate in hashing/equality.
        self.preagg: Optional["PreAggSpec"] = None
        self.stack_depth: int = stack_depth
        self.entity: Optional[Entity] = entity
        self.parents: Optional[List[Feature]] = (
            parents  # Which are the parent variables
        )
        self.intervals: List[str] = (
            intervals or []
        )  # Do we care about some past time intervals?
        self.specials: List[Any] = specials or []  # Do we care about specific values?
        self.sort: Optional[str] = sort  # Sort by...
        self.description: str = description
        # Optional role -> value map (e.g. {"a": "order", "b": "deliver",
        # "terminal": "cancel"}) used by predicate-driven aggregators.
        self.predicates: Dict[str, str] = (
            predicates if isinstance(predicates, dict) else {}
        )

    def __repr__(self) -> str:
        return f"""Feature({self.name.replace('"', "")})"""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return False
        return self.name == other.name

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.type) ^ hash((self.name, self.type))

    @property
    def query(self) -> str:
        return f"""{self.definition} as "{str.replace(self.name, '"', "")}" """

    @property
    def short_name(self) -> str:
        """A deterministic, PostgreSQL-safe bare identifier for this feature.

        Feature names become Parquet column names and persisted
        feature-importance keys downstream, so this must be stable across runs
        and processes. Long names are truncated with a stable hash suffix (the
        same scheme as :func:`pg_identifier`, minus the quoting); short names
        pass through unchanged. Never use ``hash()`` here -- Python string
        hashing is salted per process via ``PYTHONHASHSEED`` (bug #5).
        """
        return _truncate_identifier(self.name)


class Variable(Feature):
    """Represents a column in a table.

    ``role`` (optional) declares how a *direct* variable on an entity should be
    treated by the planner, independent of its storage ``type``:

    - ``identifier`` -- excluded from feature output (e.g. a name, exact
      address, license number); featurizer is automatic and exhaustive, so the
      omission is loud, not silent.
    - ``categorical`` -- one-hot encoded against a fixed, declared vocabulary
      (``vocabulary``) or the column's PostgreSQL ``ENUM`` labels. Featurizer is
      split-blind and never learns a vocabulary by scanning data.
    - ``numeric`` -- passed through unchanged (today's default behaviour).

    ``vocabulary`` is the declared, fixed set of category values for a
    ``categorical`` role. When absent, the value is resolved from the column's
    PostgreSQL ``ENUM`` at construction time (see ``featurizer.categoricals``).
    """

    def __init__(
        self,
        name: str,
        type: str,
        entity: Entity,
        predicates: Optional[Dict[str, str]] = None,
        role: Optional[str] = None,
        vocabulary: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name=name,
            definition=name,
            type=type,
            entity=entity,
            stack_depth=0,
            predicates=predicates,
        )
        self.role: Optional[str] = role
        # Resolved at construction (declared list or introspected ENUM labels),
        # always sorted for deterministic one-hot column order. None until
        # resolved / not applicable.
        self.vocabulary: Optional[List[str]] = list(vocabulary) if vocabulary else None


class Id(Feature):
    """Represents an entity id"""

    def __init__(self, name: str, entity: Entity) -> None:
        super().__init__(name=name, definition=name, type="index", entity=entity)


class Key(Feature):
    """Represents a reference to another table"""

    def __init__(self, name: str, entity: Entity) -> None:
        super().__init__(name=name, definition=name, type="key", entity=entity)


class EdgeSpec:
    """Graph-edge metadata declared on an edge-table entity.

    The edge table has a ``source`` and ``target`` node-id column, and optionally
    a ``weight`` and a ``timestamp`` (used for the causal bound, so degree is
    measured as-of each cutoff). ``node`` is the alias of the node entity these
    edges connect; the planner attaches graph features to that node.
    """

    def __init__(
        self,
        entity: "Entity",
        node: str,
        source: str,
        target: str,
        weight: Optional[str] = None,
        timestamp: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> None:
        self.entity: "Entity" = entity
        self.alias: str = entity.alias
        self.table: str = entity.table
        self.node: str = node  # node entity alias
        self.source: str = source
        self.target: str = target
        self.weight: Optional[str] = weight
        self.timestamp: Optional[str] = timestamp
        # Requested graph feature families; ``degree`` keeps the historical
        # behaviour when the config does not ask for anything else.
        self.features: List[str] = list(features) if features else ["degree"]

    def __repr__(self) -> str:
        return f"EdgeSpec({self.alias}: {self.source}->{self.target} on {self.node})"


class PeerGroupSpec:
    """Peer-group metadata declared on an entity via ``peer_groups``.

    Peers of a row are the other rows of the *same* entity that share the value
    of the categorical column ``by`` (e.g. facilities of the same
    ``facility_type``). The planner emits leave-one-out, as-of-bounded features
    comparing each ego to its peer set: group size, the mean per-peer event
    count over a child stream, and — per numeric ``measures`` column — the peer
    mean / z-score / percentile and the ego-minus-peer-mean delta. ``measures``
    defaults (at plan time) to the entity's numeric variables.
    """

    def __init__(
        self,
        entity: "Entity",
        by: str,
        measures: Optional[List[str]] = None,
    ) -> None:
        self.entity: "Entity" = entity
        self.by: str = by
        # ``None`` means "default to the entity's numeric variables" (resolved
        # by the planner); an explicit (possibly empty) list is honoured as-is.
        self.measures: Optional[List[str]] = (
            list(measures) if measures is not None else None
        )

    def __repr__(self) -> str:
        return f"PeerGroupSpec({self.entity.alias} by {self.by})"


class SpatialIx:
    """Location metadata for an entity's events.

    Either a ``{lat, lon}`` pair (plain-SQL haversine path) or a single
    ``geom`` column (PostGIS path). Component columns are exposed as ``columns``
    (index features) so they flow through the synth/transform CTEs and are
    readable by spatial aggregators.
    """

    def __init__(
        self,
        entity: Entity,
        lat: Optional[str] = None,
        lon: Optional[str] = None,
        geom: Optional[str] = None,
        srid: int = 4326,
    ) -> None:
        self.entity: Entity = entity
        self.lat: Optional[str] = lat
        self.lon: Optional[str] = lon
        self.geom: Optional[str] = geom
        self.srid: int = srid
        self.backend: str = "postgis" if geom else "plain"
        cols = [geom] if geom else [c for c in (lat, lon) if c]
        self.columns: List[Id] = [Id(name=c, entity=entity) for c in cols]

    def __repr__(self) -> str:
        if self.geom:
            return f"SpatialIx(geom={self.geom})"
        return f"SpatialIx(lat={self.lat}, lon={self.lon})"


#: Spatial second-table feature families. ``colocation_count`` counts ``right``
#: rows within ``within_m`` of each ``left`` ego; ``distance_to_nearest`` is the
#: minimum great-circle distance; ``kde_intensity`` sums a Gaussian kernel
#: (bandwidth ``bandwidth_m``) over the in-radius neighbours.
SPATIAL_FEATURE_FAMILIES = (
    "colocation_count",
    "distance_to_nearest",
    "kde_intensity",
)


class SpatialRelationshipSpec:
    """A spatial second-table relationship declared at config top level.

    For each ``left`` entity row (the ego), features are computed over the
    ``right`` entity's rows that fall within ``within_m`` metres of the ego's
    location. Both entities must declare a plain ``{lat, lon}`` ``spatial_ix``.
    When ``right`` has a ``temporal_ix`` the neighbour scan is bounded
    ``<= aod.as_of_date`` so co-location is measured as-of each cutoff. When
    ``left`` and ``right`` are the same entity the ego is excluded from its own
    neighbourhood.
    """

    def __init__(
        self,
        name: str,
        left: str,
        right: str,
        within_m: float,
        bandwidth_m: Optional[float] = None,
        features: Optional[List[str]] = None,
    ) -> None:
        self.name: str = name
        self.left: str = left
        self.right: str = right
        self.within_m: float = within_m
        # KDE bandwidth defaults to the search radius.
        self.bandwidth_m: float = bandwidth_m if bandwidth_m is not None else within_m
        self.features: List[str] = (
            list(features) if features else list(SPATIAL_FEATURE_FAMILIES)
        )

    def __repr__(self) -> str:
        return f"SpatialRelationshipSpec({self.name}: {self.left} near {self.right})"
