# coding: utf-8

"""Column-group sharding for wide feature matrices (issue #7).

PostgreSQL caps a result/CTE target list at **1664 entries**
(``target lists can have at most 1664 entries``) and a table at 1600 columns.
A wide featurizer config (many child variables × aggregations × intervals ×
transformers) blows past this, and crucially the wide ``<target>_transform``
CTE is itself a tuple, so the limit bites at the *CTE boundary* — sharding only
the outer ``select *`` is not enough.

This module partitions the target's feature columns into ordered groups and,
**for each group**, rebuilds a self-contained, valid query in which no
intermediate CTE tuple exceeds the limit either:

* the target ``transform`` CTE projects only that group's feature columns
  (plus the carried identifier columns), and
* the upstream ``synth`` and per-child ``aggs`` CTEs are pruned to only the
  columns feeding that group, dropping the joins / CTEs nothing in the group
  needs.

All groups carry ``as_of_date`` and the target id, so the artifacts re-join on
``(as_of_date, id)`` to reproduce the full matrix (the triage-pg feature-group
pattern). See ``docs/adr/0005-column-group-sharding.md``.

The structured metadata this consumes (``ShardableCTE`` / ``ColumnSpec`` /
``synth_column_source``) is recorded by the planner; this module never parses a
rendered CTE string for its column lists, only scans CTE/join *text* for CTE
*names* during the reachability pass (names are unique identifiers).
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set

from loguru import logger

from .planner import ColumnSpec, PlannerResult, ShardableCTE

# PostgreSQL's hard limit on entries in a result/CTE target list. A table is
# capped slightly lower at 1600 columns; we budget against the stricter table
# limit so an emitted group could also be materialized into a table.
PG_MAX_TARGET_LIST = 1664
PG_MAX_TABLE_COLUMNS = 1600

# Feature columns per group. Left well under the 1600 table limit so the two
# always-present carry columns (``as_of_date`` + the target id) and any extra
# carried index columns never push a group over.
DEFAULT_MAX_COLUMNS_PER_GROUP = 1400


@dataclass(frozen=True)
class GroupedQueries:
    """Result of sharding: the ordered group queries plus the join keys.

    ``queries`` maps ``group_<NNN>`` -> the self-contained SQL for that group.
    ``key_columns`` are the leading columns every group projects and on which
    all groups re-join (``["as_of_date", <target id>]``). ``fits_single`` is
    True when the whole matrix fits in one group (the single-query fast path).
    """

    queries: "OrderedDict[str, str]"
    key_columns: List[str]
    fits_single: bool


def _cte_name_scanner(names: List[str]) -> "re.Pattern[str]":
    """Compile one regex that finds any of ``names`` on word boundaries.

    CTE names are bare SQL identifiers (e.g. ``orders_aggs_for_customers``);
    word-boundary matching avoids a name matching inside a longer identifier.
    Longer names are tried first so a prefix name does not shadow a longer one.
    """
    ordered = sorted(set(names), key=len, reverse=True)
    alternation = "|".join(re.escape(n) for n in ordered)
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{alternation})(?![A-Za-z0-9_])")


class ColumnGroupSharder:
    """Partitions a :class:`PlannerResult` into joinable column-group queries."""

    def __init__(
        self,
        plan: PlannerResult,
        *,
        max_columns_per_group: int = DEFAULT_MAX_COLUMNS_PER_GROUP,
    ) -> None:
        if max_columns_per_group < 1:
            raise ValueError("max_columns_per_group must be a positive integer.")
        self.plan = plan
        self.max_columns_per_group = max_columns_per_group

        self.target_alias = plan.target.alias
        self.transform_name = f"{self.target_alias}_transform"
        self.synth_name = f"{self.target_alias}_synth"

        transform_spec = plan.cte_specs.get(self.transform_name)
        synth_spec = plan.cte_specs.get(self.synth_name)
        if transform_spec is None or synth_spec is None:
            raise ValueError(
                "Planner did not record sharding metadata for the target's "
                f"synth/transform CTEs ({self.synth_name!r} / "
                f"{self.transform_name!r}). The plan must come from "
                "FeaturePlanner.plan(); sharding cannot proceed."
            )
        self.transform_spec: ShardableCTE = transform_spec
        self.synth_spec: ShardableCTE = synth_spec

        # Identifier columns the final query carries (as_of_date is added by the
        # wrapper). The target id is the first identifier column.
        self._id_columns = list(self.synth_spec.key_columns)
        self._key_columns = ["as_of_date"] + [
            self._bare(name) for name in self._id_columns
        ]

        # Agg CTEs that feed the *target's* synth — the only aggs that are pruned
        # per group (their columns are target synth columns). Deeper-chain agg
        # CTEs (e.g. ``items_aggs_for_orders`` when ``orders`` is not the target)
        # feed a *child* synth, carry no target synth columns, and so must be
        # emitted whole. Identified by being the source CTE of a target synth
        # column with ``kind == "aggs"``.
        self._target_agg_ctes: Set[str] = {
            cte_name
            for cte_name, _ in plan.synth_column_source.values()
            if plan.cte_specs.get(cte_name) is not None
            and plan.cte_specs[cte_name].kind == "aggs"
        }

        # All registered CTE names (for the reachability name scan).
        self._all_cte_names = list(plan.cte_order)
        self._name_scanner = (
            _cte_name_scanner(self._all_cte_names) if self._all_cte_names else None
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def fits_single_group(self) -> bool:
        """True when the unsharded query is valid on PostgreSQL.

        This is the test ``Featurizer.query`` uses, judged against the *hard*
        PostgreSQL limit (``PG_MAX_TARGET_LIST``), independent of the (smaller,
        headroom-leaving) ``max_columns_per_group`` partition size: a single
        valid query requires the target transform tuple **and** every
        intermediate CTE tuple to be ≤ the limit. A 1500-column matrix is one
        valid query even though it would shard into two groups for output.
        """
        target_width = len(self.transform_spec.key_columns) + len(
            self.transform_spec.columns
        )
        return (
            target_width <= PG_MAX_TARGET_LIST
            and not self._oversized_intermediate_ctes()
        )

    @property
    def n_groups(self) -> int:
        """Number of column groups the target transform partitions into."""
        n_cols = len(self.transform_spec.columns)
        if n_cols == 0:
            return 1
        per = self.max_columns_per_group
        return (n_cols + per - 1) // per

    def group_queries(self) -> "OrderedDict[str, str]":
        """Render every column group to a self-contained SQL string."""
        return self.build().queries

    def build(self) -> GroupedQueries:
        """Partition + render. Returns the ordered group queries and join keys."""
        groups = self._partition_columns()
        queries: "OrderedDict[str, str]" = OrderedDict()
        for idx, group_columns in enumerate(groups):
            gid = f"group_{idx:03d}"
            queries[gid] = self._render_group(group_columns)
        return GroupedQueries(
            queries=queries,
            key_columns=list(self._key_columns),
            fits_single=self.fits_single_group,
        )

    # ------------------------------------------------------------------ #
    # Partitioning
    # ------------------------------------------------------------------ #

    def _partition_columns(self) -> List[List[ColumnSpec]]:
        """Split the target transform columns into ordered ≤-limit groups."""
        columns = list(self.transform_spec.columns)
        if not columns:
            return [[]]
        per = self.max_columns_per_group
        return [columns[i : i + per] for i in range(0, len(columns), per)]

    # ------------------------------------------------------------------ #
    # Per-group rendering
    # ------------------------------------------------------------------ #

    def _render_group(self, group_columns: List[ColumnSpec]) -> str:
        """Build the self-contained query for one column group."""
        # 1. Synth columns this group needs = union of its columns' deps.
        needed_synth: Set[str] = set()
        for col in group_columns:
            needed_synth.update(col.depends_on)

        # A pathological fan-out (transformers referencing many synth columns)
        # could push the pruned synth tuple over the limit even though the
        # transform tuple is within budget. Fail fast with context rather than
        # emit a query PostgreSQL will reject.
        synth_width = len(self.synth_spec.key_columns) + len(needed_synth)
        if synth_width > PG_MAX_TARGET_LIST:
            raise ValueError(
                f"A column group's pruned synth CTE would project {synth_width} "
                f"columns, over PostgreSQL's {PG_MAX_TARGET_LIST}-entry limit: the "
                f"group's {len(group_columns)} transform columns depend on "
                f"{len(needed_synth)} distinct synth columns. Lower "
                "max_columns_per_group so each group's synth fan-out stays under "
                "the limit."
            )

        # 2. Joins + upstream CTEs feeding those synth columns. Base-table
        #    variables have no source entry (they come from the target table).
        kept_joins: List[str] = []
        seen_joins: Set[str] = set()
        kept_upstream: Set[str] = set()
        for col in sorted(needed_synth):
            source = self.plan.synth_column_source.get(col)
            if source is None:
                continue  # base-table variable, available from the target table
            cte_name, join_sql = source
            kept_upstream.add(cte_name)
            if join_sql not in seen_joins:
                seen_joins.add(join_sql)
                kept_joins.append(join_sql)

        # 3. Reachability: pull in every CTE transitively referenced by the
        #    target synth/transform, the kept upstream CTEs, or the kept joins.
        reachable = self._reachable_ctes(kept_upstream, kept_joins)

        # 4. Render the CTE list in original emission order.
        rendered_ctes = self._render_ctes(
            reachable, group_columns, needed_synth, kept_joins
        )

        return self._wrap(rendered_ctes, group_columns)

    def _reachable_ctes(
        self, kept_upstream: Set[str], kept_joins: List[str]
    ) -> Set[str]:
        """Transitive closure of CTE names a group's query references.

        Seeds: target synth + transform (always present), the upstream CTEs
        feeding kept synth columns, and any CTE named in a kept join's text
        (e.g. an as-of lateral join reads ``<source>_transform``). The frontier
        then expands by scanning each *non-target* reached CTE's own body for
        further CTE names.

        The target synth/transform are deliberately **not** body-scanned: they
        are re-rendered per group with pruned joins, so their dependencies are
        exactly ``kept_upstream`` + ``kept_joins`` — scanning their full-width
        ``rendered`` text would spuriously pull in agg/peer CTEs this group
        dropped.
        """
        target_ctes = {self.synth_name, self.transform_name}
        reachable: Set[str] = set(target_ctes)

        seeds: Set[str] = {n for n in kept_upstream if n in self.plan.cte_order}
        for join_sql in kept_joins:
            seeds.update(self._names_in(join_sql))
        seeds -= target_ctes

        frontier = list(seeds)
        reachable.update(seeds)
        while frontier:
            name = frontier.pop()
            body = self._cte_body(name)
            if body is None:
                continue
            for ref in self._names_in(body):
                if ref not in reachable and ref not in target_ctes:
                    reachable.add(ref)
                    frontier.append(ref)
        return reachable

    def _render_ctes(
        self,
        reachable: Set[str],
        group_columns: List[ColumnSpec],
        needed_synth: Set[str],
        kept_joins: List[str],
    ) -> List[str]:
        """Render each reachable CTE in emission order, pruning the target's."""
        out: List[str] = []
        for name in self.plan.cte_order:
            if name not in reachable:
                continue
            if name == self.transform_name:
                out.append(self._render_transform(group_columns))
            elif name == self.synth_name:
                out.append(self._render_synth(needed_synth, kept_joins))
            elif name in self._target_agg_ctes:
                # A target-level agg CTE: prune to the surviving target synth
                # columns it feeds.
                out.append(self._render_agg(self.plan.cte_specs[name], needed_synth))
            else:
                # A bounded / non-target CTE (verbatim string, a deeper-chain
                # agg, or a full-width child synth/transform): emit whole.
                out.append(self._cte_body(name) or "")
        return out

    def _render_transform(self, group_columns: List[ColumnSpec]) -> str:
        spec = self.transform_spec
        projections = list(spec.key_columns) + [c.projection for c in group_columns]
        return spec.prefix + ",\n        ".join(projections) + spec.suffix

    def _render_synth(self, needed_synth: Set[str], kept_joins: List[str]) -> str:
        spec = self.synth_spec
        surviving = [c.projection for c in spec.columns if c.name in needed_synth]
        projections = list(spec.key_columns) + surviving
        select_list = ",\n        ".join(projections)
        joins_sql = ""
        if kept_joins:
            joins_sql = "\n        left join " + "\n        left join ".join(kept_joins)
        return (
            spec.prefix
            + select_list
            + spec.suffix
            + joins_sql
            + "\n        )\n        "
        )

    def _render_agg(self, spec: ShardableCTE, needed_synth: Set[str]) -> str:
        surviving = [c.projection for c in spec.columns if c.name in needed_synth]
        # The group only reaches this agg CTE because it keeps ≥1 of its
        # columns, so ``surviving`` is non-empty here.
        projections = list(spec.key_columns) + surviving
        return spec.prefix + ",\n        ".join(projections) + spec.suffix

    def _wrap(self, rendered_ctes: List[str], group_columns: List[ColumnSpec]) -> str:
        """Wrap the CTEs in the lateral-join shell, selecting the group output.

        Identical shape to :class:`featurizer.sql.SQLRenderer`: ``t.*`` over the
        pruned ``<target>_transform`` CTE, which is built to lead with the
        identifier columns (the target id first) followed by this group's
        feature columns — so every group's output is ``(as_of_date, <id>,
        <group features…>)`` and the groups re-join on ``(as_of_date, id)``.

        ``t.*`` (not an explicit ``t."col"`` list) is deliberate: generated
        feature identifiers can exceed PostgreSQL's 63-byte limit and a verbatim
        long name would not resolve against the (server-truncated) projected
        column. The transform CTE already fixes the column order, so ``*`` is
        both correct and limit-safe.
        """
        ctes = ",".join(rendered_ctes)
        return f"""
        select aod.as_of_date, t.*
        from as_of_dates as aod
        cross join lateral (

        with

        {ctes}

        select * from {self.transform_name}
        ) as t

        order by aod.as_of_date
        """

    # ------------------------------------------------------------------ #
    # Limit diagnostics
    # ------------------------------------------------------------------ #

    def _oversized_intermediate_ctes(self) -> Dict[str, int]:
        """CTEs that sharding cannot shrink yet still exceed the limit.

        These are the *documented limitation*: a single child entity whose own
        transform/synth tuple (or a deeper-chain agg CTE) already exceeds 1664
        cannot be made to fit by sharding the *target's* output — pruning
        operates per target column group, and such a CTE is reused whole across
        groups. The target's own ``transform``/``synth`` and its per-child
        ``aggs`` are pruned per group, so they are *not* counted here even when
        their full width is over the limit. We surface the rest rather than
        silently truncating.
        """
        prunable = {self.transform_name, self.synth_name} | self._target_agg_ctes
        oversized: Dict[str, int] = {}
        for name, spec in self.plan.cte_specs.items():
            if name in prunable:
                continue  # re-rendered (pruned) per group, never over-limit there
            width = len(spec.key_columns) + len(spec.columns)
            if width > PG_MAX_TARGET_LIST:
                oversized[name] = width
        return oversized

    def warn_oversized(self) -> None:
        """Log a clear warning for any intermediate CTE still over the limit."""
        for name, width in self._oversized_intermediate_ctes().items():
            logger.warning(
                "Sharding bound: intermediate CTE {!r} projects {} columns, over "
                "PostgreSQL's {}-entry target-list limit. Column-group sharding "
                "splits the *target* output but reuses this CTE whole, so groups "
                "referencing it may still be rejected. Reduce the child entity's "
                "primitive/interval breadth, or raise the relationship that "
                "produces it to the target so its columns can be grouped.",
                name,
                width,
                PG_MAX_TARGET_LIST,
            )

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _cte_body(self, name: str) -> str | None:
        """The full-width rendered text for a CTE (verbatim or shardable).

        Non-target CTEs are emitted byte-for-byte identical to the single-query
        renderer (the planner stashed each one's full text), so a non-target
        child synth keeps its own joins and a single-group config reproduces
        ``Featurizer.query`` exactly. Only the *target's* synth/transform/aggs
        are re-rendered (pruned) per group, handled by the callers.
        """
        if name in self.plan.verbatim_ctes:
            return self.plan.verbatim_ctes[name]
        spec = self.plan.cte_specs.get(name)
        if spec is None:
            return None
        return spec.rendered

    def _names_in(self, text: str) -> Set[str]:
        if self._name_scanner is None:
            return set()
        return set(self._name_scanner.findall(text))

    @staticmethod
    def _bare(name: str) -> str:
        """The unqualified column reference for ``t.<col>`` selection.

        ``key_columns`` arrive table-qualified (``customers.customer_id``); the
        outer select references them by their projected (bare) name. Quoted
        feature identifiers are passed through unchanged.
        """
        if name.startswith('"'):
            return name
        return name.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------- #
# Temp-table materialization for oversized NON-TARGET child CTEs (issue #7).
#
# Column-group sharding splits the *target's* output, but a non-target child
# CTE (a deeper-chain ``<src>_aggs_for_<parent>`` or a child's own
# ``<child>_synth`` / ``<child>_transform``) is reused whole across groups, so a
# single child CTE wider than 1664 cannot be made to fit by grouping the target.
# The cascade is inherent: an oversized child agg forces its consumer synth (and
# then transform) over the limit too, since the synth projects every agg column.
#
# The fix materializes each oversized child CTE into keyed TEMP-table *shards*
# (each ≤ the column budget), bottom-up, and rewrites every downstream reference
# from an inline CTE into joins against the shards. Execution becomes a
# session-scoped sequence: run the ``CREATE TEMP TABLE … ON COMMIT DROP AS …``
# preamble, then the target column-group SELECT(s), all on one connection.
# ---------------------------------------------------------------------------- #

# Prefix namespacing featurizer's temp tables away from user/session objects.
TEMP_TABLE_PREFIX = "__fz_"
# Feature columns per shard. Same budget as a column group: leaves headroom
# under the 1600-column table limit for the carried key column(s).
MATERIALIZE_MAX_COLUMNS = DEFAULT_MAX_COLUMNS_PER_GROUP

_TRANSFORM_EGO_ALIAS = "_ego"  # matches planner.TRANSFORM_EGO_ALIAS


@dataclass(frozen=True)
class MaterializedShard:
    """One TEMP-table shard of an oversized CTE.

    ``cte_name`` is the logical CTE being materialized; ``table_name`` is the
    shard's TEMP table; ``join_key`` is the bare key column it projects and that
    its consumers join on; ``columns`` are the feature column names in this shard
    (excluding the key); ``create_sql`` is the full ``CREATE TEMP TABLE … AS …``.
    """

    cte_name: str
    table_name: str
    join_key: str
    columns: List[str]
    create_sql: str


@dataclass(frozen=True)
class MaterializationPlan:
    """Result of materialization planning.

    ``ddl`` is the ordered list of ``CREATE TEMP TABLE`` statements to run as a
    preamble (bottom-up: an oversized upstream before its consumer).
    ``shards_by_cte`` maps each materialized CTE name to its ordered shards.
    ``materialized_ctes`` is the set of logical CTE names now backed by temp
    tables — the group-query renderer drops these from its ``with`` lists and
    joins their shards instead.
    """

    ddl: List[str]
    shards_by_cte: "OrderedDict[str, List[MaterializedShard]]"
    materialized_ctes: Set[str] = field(default_factory=set)


class MaterializationPlanner:
    """Plans TEMP-table materialization for a plan's oversized child CTEs.

    Self-sufficient: derives everything it needs (the target's CTE names, the
    prunable target-level agg set, a CTE-name scanner, and CTE bodies) directly
    from the :class:`PlannerResult`, so it shares no private state with
    :class:`ColumnGroupSharder`. ``materialize_threshold`` defaults to the hard
    PostgreSQL limit; tests pass a small value to force materialization on a
    small config (mirroring ``ColumnGroupSharder.max_columns_per_group``) without
    building a genuinely 1664-wide CTE.
    """

    def __init__(
        self,
        plan: PlannerResult,
        *,
        materialize_threshold: int = PG_MAX_TARGET_LIST,
        max_columns_per_shard: int = MATERIALIZE_MAX_COLUMNS,
    ) -> None:
        if materialize_threshold < 1 or max_columns_per_shard < 1:
            raise ValueError(
                "materialize_threshold and max_columns_per_shard must be positive."
            )
        self.plan = plan
        self.materialize_threshold = materialize_threshold
        self.max_columns_per_shard = max_columns_per_shard

        self.target_alias = plan.target.alias
        self.transform_name = f"{self.target_alias}_transform"
        self.synth_name = f"{self.target_alias}_synth"
        # Target-level aggs are pruned per group, never materialized (same rule
        # as ColumnGroupSharder): a synth-column source CTE of kind "aggs".
        self._target_agg_ctes: Set[str] = {
            cte_name
            for cte_name, _ in plan.synth_column_source.values()
            if plan.cte_specs.get(cte_name) is not None
            and plan.cte_specs[cte_name].kind == "aggs"
        }
        self._scanner = _cte_name_scanner(plan.cte_order) if plan.cte_order else None

    # ------------------------------------------------------------------ #
    # Detection + ordering
    # ------------------------------------------------------------------ #

    def oversized_ctes(self) -> Dict[str, int]:
        """Non-prunable intermediate CTEs wider than the threshold.

        Same prunable set as :meth:`ColumnGroupSharder._oversized_intermediate_ctes`
        (target transform/synth + target-level aggs are pruned per group, never
        materialized), but judged against the configurable threshold.
        """
        prunable = {self.transform_name, self.synth_name} | self._target_agg_ctes
        out: Dict[str, int] = {}
        for name, spec in self.plan.cte_specs.items():
            if name in prunable:
                continue
            width = len(spec.key_columns) + len(spec.columns)
            if width > self.materialize_threshold:
                out[name] = width
        return out

    def materialization_order(self) -> List[str]:
        """Oversized CTEs in bottom-up order (an upstream before its consumer).

        Edges are discovered by scanning each oversized CTE's body for the names
        of *other* oversized CTEs (the existing name-reachability scan), never by
        parsing column lists. The planner DAG is acyclic by construction
        (children are built before parents), so a stable topological sort exists.
        """
        oversized = set(self.oversized_ctes())
        # deps[a] = oversized CTEs whose names appear in a's body (a's upstreams).
        deps: Dict[str, Set[str]] = {}
        for name in oversized:
            body = self._body(name) or ""
            refs = self._scan(body) & oversized
            refs.discard(name)
            deps[name] = refs

        ordered: List[str] = []
        placed: Set[str] = set()
        # Emit in the planner's original CTE order, deferring any CTE whose
        # oversized upstreams have not all been placed (a simple, stable
        # Kahn-style pass; the DAG is small and acyclic).
        remaining = [n for n in self.plan.cte_order if n in oversized]
        # Any oversized CTE not in cte_order (defensive) goes last.
        remaining += [n for n in oversized if n not in remaining]
        progress = True
        while remaining and progress:
            progress = False
            still: List[str] = []
            for name in remaining:
                if deps[name] <= placed:
                    ordered.append(name)
                    placed.add(name)
                    progress = True
                else:
                    still.append(name)
            remaining = still
        if remaining:
            raise ValueError(
                "Cycle detected among oversized CTEs while planning "
                f"materialization: {remaining}. The planner DAG should be acyclic."
            )
        return ordered

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self) -> MaterializationPlan:
        """Plan the full preamble: every oversized CTE's shard DDL, bottom-up."""
        order = self.materialization_order()
        shards_by_cte: "OrderedDict[str, List[MaterializedShard]]" = OrderedDict()
        ddl: List[str] = []
        materialized: Set[str] = set()
        for cte_name in order:
            shards = self._shards_for(cte_name, shards_by_cte)
            shards_by_cte[cte_name] = shards
            ddl.extend(shard.create_sql for shard in shards)
            materialized.add(cte_name)
        return MaterializationPlan(
            ddl=ddl,
            shards_by_cte=shards_by_cte,
            materialized_ctes=materialized,
        )

    # ------------------------------------------------------------------ #
    # Per-CTE shard construction
    # ------------------------------------------------------------------ #

    def _shards_for(
        self,
        cte_name: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> List[MaterializedShard]:
        spec = self.plan.cte_specs[cte_name]
        mkey = self.plan.materialization_keys.get(cte_name)
        if mkey is None:
            raise ValueError(
                f"Cannot materialize CTE {cte_name!r}: the planner recorded no "
                "join key for it (an id-less entity cannot be re-joined). Narrow "
                "this entity's primitive/interval breadth instead."
            )
        join_key = mkey.join_key
        column_chunks = self._partition(spec.columns)
        shards: List[MaterializedShard] = []
        for idx, chunk in enumerate(column_chunks):
            table_name = self._temp_name(cte_name, idx)
            select_body = self._select_body(spec, chunk, idx, join_key, done)
            create_sql = (
                f"create temp table {table_name} on commit drop as\n{select_body}"
            )
            shards.append(
                MaterializedShard(
                    cte_name=cte_name,
                    table_name=table_name,
                    join_key=join_key,
                    columns=[c.name for c in chunk],
                    create_sql=create_sql,
                )
            )
        return shards

    def _select_body(
        self,
        spec: ShardableCTE,
        chunk: List[ColumnSpec],
        idx: int,
        join_key: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> str:
        if spec.kind == "aggs":
            return self._agg_select(spec, chunk, done)
        if spec.kind == "synth":
            return self._synth_select(spec, chunk, idx, join_key, done)
        if spec.kind == "transform":
            return self._transform_select(spec, chunk, idx, join_key, done)
        raise ValueError(f"Cannot materialize CTE of kind {spec.kind!r}.")

    def _agg_select(
        self,
        spec: ShardableCTE,
        chunk: List[ColumnSpec],
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> str:
        """``select <key>, <chunk> from <src>_transform [where] group by <key>``.

        The agg groups child rows by the parent key (already in ``key_columns``),
        so every shard projects that single key. Its source ``<src>_transform`` is
        reused as-is when bounded (pulled into an inline ``with``) or rewritten to
        its shards when it was itself materialized.
        """
        projections = list(spec.key_columns) + [c.projection for c in chunk]
        tail = self._strip_cte_close(spec.suffix)
        tail = self._rewrite_from_sources(tail, done)
        with_clause = self._inline_with(spec.name, done)
        return with_clause + "select\n        " + ",\n        ".join(projections) + tail

    def _synth_select(
        self,
        spec: ShardableCTE,
        chunk: List[ColumnSpec],
        idx: int,
        join_key: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> str:
        """``select <keys>, <chunk> from <table> <rewritten left joins>``.

        Only the first shard carries the full identifier columns; later shards
        carry just the join key, so re-joining the shards with ``using(<key>)``
        yields each non-key column exactly once. Child-agg joins that were
        materialized are expanded into per-shard ``left join``s.
        """
        keys = self._shard_keys(spec, idx, join_key)
        projections = keys + [c.projection for c in chunk]
        entity = self._entity_of(spec)
        joins = self.plan.joins.get(entity, [])
        joins_sql = "".join(
            "\n        left join " + j for j in self._rewrite_joins(joins, done)
        )
        with_clause = self._inline_with(spec.name, done, skip_joins=True)
        return (
            with_clause
            + "select\n        "
            + ",\n        ".join(projections)
            + spec.suffix
            + joins_sql
        )

    def _transform_select(
        self,
        spec: ShardableCTE,
        chunk: List[ColumnSpec],
        idx: int,
        join_key: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> str:
        """``select <keys>, <chunk> from <synth source> _ego``.

        The transform reads one row per entity from its synth; when that synth was
        materialized, its shards are re-joined (``using(<key>)``) into a subquery
        aliased ``_ego`` so the transformer projections still resolve their synth
        columns by bare name.
        """
        keys = self._shard_keys(spec, idx, join_key)
        projections = keys + [c.projection for c in chunk]
        synth_name = f"{self._entity_of(spec)}_synth"
        source = self._from_source(synth_name, _TRANSFORM_EGO_ALIAS, done)
        return (
            "select\n        "
            + ",\n        ".join(projections)
            + "\n        from "
            + source
        )

    # ------------------------------------------------------------------ #
    # Rewriting helpers
    # ------------------------------------------------------------------ #

    def _rewrite_joins(
        self,
        joins: List[str],
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> List[str]:
        """Expand each ``left join`` clause that targets a materialized CTE into
        one clause per shard (the CTE name swapped for each shard table)."""
        out: List[str] = []
        for join in joins:
            hit = next((c for c in done if self._mentions(join, c)), None)
            if hit is None:
                out.append(join)
                continue
            for shard in done[hit]:
                out.append(join.replace(hit, shard.table_name))
        return out

    def _rewrite_from_sources(
        self,
        text: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> str:
        """Rewrite ``from <cte>`` where ``<cte>`` was materialized into a re-join
        subquery over its shards (used by the agg path for a materialized source
        transform on deeper chains)."""
        for cte in done:
            pattern = re.compile(rf"from\s+{re.escape(cte)}(?![A-Za-z0-9_])")
            if pattern.search(text):
                rejoin = self._rejoin_subquery(done[cte])
                text = pattern.sub(f"from {rejoin} {cte}", text)
        return text

    def _from_source(
        self,
        cte_name: str,
        alias: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
    ) -> str:
        """The FROM source for a CTE read directly (``from <cte> <alias>``):
        the re-joined shards when materialized, else the CTE name."""
        if cte_name in done:
            return f"{self._rejoin_subquery(done[cte_name])} {alias}"
        return f"{cte_name} {alias}"

    def _rejoin_subquery(self, shards: List[MaterializedShard]) -> str:
        """``(select * from s0 left join s1 using(<key>) …)`` — re-joins shards
        into one logical row per key. Only the first shard carries the non-key
        identifier columns, so ``select *`` yields each column once."""
        if not shards:
            return "(select 1)"
        key = shards[0].join_key
        first = shards[0].table_name
        joins = "".join(f" left join {s.table_name} using ({key})" for s in shards[1:])
        return f"(select * from {first}{joins})"

    def _inline_with(
        self,
        cte_name: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
        *,
        skip_joins: bool = False,
    ) -> str:
        """A ``with`` clause defining the non-materialized CTEs a materialized
        CTE's body needs (e.g. an agg's ``<src>_transform`` + ``<src>_synth``),
        in planner emission order. Materialized upstreams are excluded — they are
        temp tables, referenced by the rewrite, not redefined here."""
        needed = self._inline_upstreams(cte_name, done, skip_joins=skip_joins)
        if not needed:
            return ""
        bodies = [self._body(n) or "" for n in needed]
        return "with\n" + ",".join(bodies) + "\n"

    def _inline_upstreams(
        self,
        cte_name: str,
        done: "OrderedDict[str, List[MaterializedShard]]",
        *,
        skip_joins: bool,
    ) -> List[str]:
        """Transitive non-materialized CTE names a materialized CTE references,
        in planner emission order (dependencies first)."""
        spec = self.plan.cte_specs.get(cte_name)
        # Seed from the CTE's own rendered body (its FROM / joins name upstreams).
        if skip_joins and spec is not None:
            seed_text = spec.suffix
        else:
            seed_text = self._body(cte_name) or ""
        frontier = list((self._scan(seed_text) - {cte_name}) - set(done))
        if skip_joins:
            entity = self._entity_of(spec) if spec is not None else ""
            for join in self.plan.joins.get(entity, []):
                for ref in self._scan(join):
                    if ref != cte_name and ref not in done:
                        frontier.append(ref)
        reachable: Set[str] = set()
        while frontier:
            name = frontier.pop()
            if name in reachable or name in done or name == cte_name:
                continue
            if name not in self.plan.cte_specs and name not in self.plan.verbatim_ctes:
                continue  # base table, not a CTE
            reachable.add(name)
            body = self._body(name) or ""
            frontier.extend(self._scan(body) - reachable)
        return [n for n in self.plan.cte_order if n in reachable]

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _partition(self, columns: List[ColumnSpec]) -> List[List[ColumnSpec]]:
        if not columns:
            return [[]]
        per = self.max_columns_per_shard
        return [columns[i : i + per] for i in range(0, len(columns), per)]

    def _shard_keys(self, spec: ShardableCTE, idx: int, join_key: str) -> List[str]:
        """Key columns a shard projects: the first shard carries all identifier
        columns; later shards carry only the join key (so a ``using(<key>)``
        re-join keeps each non-key identifier column unambiguous)."""
        if idx == 0:
            return list(spec.key_columns)
        qualified = next(
            (k for k in spec.key_columns if self._bare(k) == join_key),
            join_key,
        )
        return [qualified]

    def _entity_of(self, spec: ShardableCTE) -> str:
        """The entity alias owning a synth/transform CTE (``orders_synth`` ->
        ``orders``)."""
        if spec.kind == "synth":
            return spec.name[: -len("_synth")]
        if spec.kind == "transform":
            return spec.name[: -len("_transform")]
        # agg: ``<src>_aggs_for_<parent>`` -> ``<src>``
        return spec.name.split("_aggs_for_")[0]

    def _temp_name(self, cte_name: str, idx: int) -> str:
        """A collision-safe, ≤63-byte TEMP table name for a shard.

        Hashes the logical CTE name when the readable form would exceed
        PostgreSQL's 63-byte identifier limit (it truncates silently, which would
        alias two distinct CTEs onto one table)."""
        readable = f"{TEMP_TABLE_PREFIX}{cte_name}__s{idx:03d}"
        if len(readable) <= 63:
            return readable
        digest = hashlib.sha1(cte_name.encode()).hexdigest()[:8]
        return f"{TEMP_TABLE_PREFIX}{digest}__s{idx:03d}"

    def _body(self, name: str) -> str | None:
        """The full-width rendered text for a CTE (verbatim or shardable),
        byte-identical to the single-query renderer; ``None`` if unknown."""
        if name in self.plan.verbatim_ctes:
            return self.plan.verbatim_ctes[name]
        spec = self.plan.cte_specs.get(name)
        return spec.rendered if spec is not None else None

    def _scan(self, text: str) -> Set[str]:
        """Every known CTE name appearing in ``text`` (word-boundary matched)."""
        if self._scanner is None:
            return set()
        return set(self._scanner.findall(text))

    @staticmethod
    def _mentions(text: str, name: str) -> bool:
        """True if ``name`` appears in ``text`` on identifier word boundaries."""
        return bool(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
        )

    @staticmethod
    def _bare(name: str) -> str:
        """The unqualified column reference (``orders.order_id`` -> ``order_id``)."""
        if name.startswith('"'):
            return name
        return name.rsplit(".", 1)[-1]

    @staticmethod
    def _strip_cte_close(suffix: str) -> str:
        """Drop the trailing ``)`` that closes a CTE definition, leaving the bare
        ``from … [where …] [group by …]`` tail for a ``create table as select``."""
        stripped = suffix.rstrip()
        if stripped.endswith(")"):
            stripped = stripped[:-1]
        return stripped
