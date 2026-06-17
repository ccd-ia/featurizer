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

import re
from collections import OrderedDict
from dataclasses import dataclass
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
