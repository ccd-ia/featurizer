# coding: utf-8

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Set

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections import OrderedDict

    from .manifest import ManifestEntry
    from .sharding import ColumnGroupSharder, FeatureGroupTable, GroupedQueries

import pandas as pd
import yaml
from icecream import ic
from loguru import logger

from .boundary import DEFAULT_BOUNDARY, AsOfBoundary
from .categoricals import ROLE_CATEGORICAL, resolve_vocabulary
from .executor import QueryExecutor, analyze_as_of_dates
from .planner import FeaturePlanner, PlannerResult
from .primitives import Entity, ERGraph, Feature, Variable
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
        self,
        config_file: str,
        *,
        debug: bool = False,
        validate: bool = True,
        materialize_threshold: int | None = None,
        connection: Any = None,
    ) -> None:
        """Initialize Featurizer from a YAML configuration file.

        Args:
            config_file: Path to YAML configuration file
            debug: Enable debug logging with icecream. Can also be set via FEATURIZER_DEBUG env var.
            validate: Run enhanced validation checks (default: True)
            materialize_threshold: Column width above which an oversized
                non-target child CTE is materialized into TEMP-table shards
                (issue #7). Defaults to PostgreSQL's hard 1664-entry limit; lower
                it to materialize earlier (advanced / testing).
            connection: Optional psycopg connection used only to resolve a
                ``role: categorical`` direct variable's vocabulary from its
                PostgreSQL ``ENUM`` labels when no ``vocabulary`` is declared in
                the config. When omitted, one is opened from ``DATABASE_URL`` /
                ``PG*`` if any categorical actually needs it; a declared
                vocabulary needs no database (``query`` / ``--show-sql`` stay
                DB-free). The connection is never used to scan data.

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is missing required keys or has invalid values,
                or a ``role: categorical`` variable has neither a declared
                ``vocabulary`` nor an introspectable PostgreSQL ``ENUM``.
        """
        config = self._load_config(config_file, validate=validate)
        self._materialize_threshold = materialize_threshold

        self._debug_enabled: bool = debug or self._env_debug_enabled()
        if self._debug_enabled:
            ic.configureOutput(prefix="[Featurizer] ", includeContext=True)

        self.max_depth: int = config["max_depth"]
        self.intervals: List[str] = config["intervals"]
        # Point-in-time boundary mode: ``inclusive`` (default, ``<=``) keeps an
        # event dated exactly on the as_of_date knowable; ``exclusive`` (``<``)
        # treats it as not-yet-knowable. Validated in ConfigValidator.
        self.as_of_boundary: AsOfBoundary = config.get(
            "as_of_boundary", DEFAULT_BOUNDARY
        )

        self.graph: ERGraph = ERGraph(
            config["entities"],
            config["relationships"],
            config.get("spatial_relationships"),
        )
        self.target: Entity = self._get_entity(config["target"])

        # Resolve fixed vocabularies for the target's role: categorical direct
        # variables (declared list, else introspected ENUM) before planning so
        # the planner stays DB-free. Fails loud if a categorical can be neither.
        self._resolve_categorical_vocabularies(connection)

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
            boundary=self.as_of_boundary,
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

    def _resolve_categorical_vocabularies(self, connection: Any) -> None:
        """Bake a fixed vocabulary onto each target ``role: categorical`` variable.

        Declared vocabularies need no database. If any categorical lacks one, a
        connection is required to read its PostgreSQL ``ENUM`` labels: the passed
        ``connection`` is used, else one is opened from ``DATABASE_URL`` / ``PG*``;
        if neither resolves, :func:`resolve_vocabulary` raises a loud, actionable
        error. The data is never scanned for distinct values (split-blind).
        """
        pending = [
            feature
            for feature in self.target.features
            if isinstance(feature, Variable) and feature.role == ROLE_CATEGORICAL
        ]
        if not pending:
            return

        needs_db = any(not var.vocabulary for var in pending)
        conn = connection
        own_connection: Any = None
        if needs_db and conn is None:
            own_connection = conn = self._maybe_env_connection()
        try:
            for var in pending:
                var.vocabulary = resolve_vocabulary(var, self.target, conn)
        finally:
            if own_connection is not None:
                own_connection.close()

    @staticmethod
    def _maybe_env_connection() -> Any:
        """A psycopg connection from the environment, or ``None`` if unconfigured.

        Mirrors the connection sourcing used by the Arrow/Parquet output paths
        (``DATABASE_URL`` / ``PG*``). Returns ``None`` rather than raising when no
        database is configured, so the caller can produce the precise
        declare-vocabulary-or-ENUM error instead of a generic connection error.
        """
        from .arrow import default_connection

        try:
            return default_connection()
        except RuntimeError:
            return None

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
    def feature_manifest(self) -> "List[ManifestEntry]":
        """Map every output column to its full, untruncated intended name.

        One row per output feature column (in output order), each carrying the
        rendered ``column`` name, the human-readable ``label`` (recovered even
        when the 63-byte identifier cap truncated the column name), a
        ``truncated`` flag, the ``kind`` (``one_hot`` | ``variable`` |
        ``derived``), the owning ``entity``, and — for one-hot columns — the
        ``source_column`` and ``value`` they encode. Useful for human/partner
        labels, plot legends, and joining readable names back onto the matrix.
        """
        from .manifest import build_feature_manifest

        return build_feature_manifest(self._plan.target_output_features)

    def manifest_dataframe(self) -> "pd.DataFrame":
        """The feature manifest as a pandas DataFrame (table / plots / joins)."""
        from .manifest import manifest_dataframe

        return manifest_dataframe(self.feature_manifest)

    @property
    def query(self) -> str:
        """Generate the single SQL query for this featurizer configuration.

        Raises:
            ValueError: If the configuration is too wide to render as one valid
                query — the ``<target>_transform`` CTE (or an intermediate CTE)
                would exceed PostgreSQL's 1664-entry target-list limit. Use
                :attr:`query_groups` / :meth:`to_parquet` / :meth:`to_arrow`
                (column-group sharding, issue #7) instead. The matrix is never
                silently truncated.
        """
        sharder = self._sharder()
        if not sharder.fits_single_group:
            n_groups = sharder.n_groups
            raise ValueError(
                f"Feature matrix for target '{self.target.alias}' is too wide for a "
                "single query: it exceeds PostgreSQL's 1664-entry target-list limit "
                f"and partitions into {n_groups} column groups. Use "
                "`.query_groups` (group_id -> SQL), `.to_parquet(dir)` (one Parquet "
                "per group), or `.to_arrow()` (list of tables); all groups re-join "
                "on (as_of_date, id). `.to_arrow`/`.to_parquet`/`.to_dataframe` also "
                "materialize any oversized child CTE automatically (issue #7). See "
                "docs/adr/0005-column-group-sharding.md."
            )
        return self._renderer.render(self._plan)

    def _sharder(self) -> "ColumnGroupSharder":
        """A ColumnGroupSharder for this plan honouring ``materialize_threshold``."""
        from .sharding import PG_MAX_TARGET_LIST, ColumnGroupSharder

        threshold = (
            self._materialize_threshold
            if self._materialize_threshold is not None
            else PG_MAX_TARGET_LIST
        )
        return ColumnGroupSharder(self._plan, materialize_threshold=threshold)

    def _grouped(self) -> "GroupedQueries":
        """The grouped queries + any temp-table materialization preamble.

        A config that fits one valid query short-circuits to a single
        ``group_000`` equal to :attr:`query` (no preamble). A wide or
        oversized-child config returns the partitioned/rewritten group queries
        and, when a child CTE had to be materialized, the preamble on
        ``GroupedQueries.materialization`` (issue #7).
        """
        from collections import OrderedDict

        from .sharding import GroupedQueries

        sharder = self._sharder()
        sharder.warn_oversized()
        if sharder.fits_single_group:
            return GroupedQueries(
                queries=OrderedDict([("group_000", self._renderer.render(self._plan))]),
                key_columns=sharder.key_columns,
                fits_single=True,
                materialization=None,
            )
        return sharder.build()

    @property
    def query_groups(self) -> "OrderedDict[str, str]":
        """SQL for each column group: ``group_<NNN>`` -> self-contained query.

        Partitions the (possibly very wide) feature matrix into ordered column
        groups, each a valid query whose every CTE tuple is under PostgreSQL's
        1664-entry limit (issue #7). A config that fits in one query returns a
        single ``group_000`` entry equal to :attr:`query`. Every group leads
        with ``(as_of_date, <target id>)`` so the groups re-join into the full
        matrix.

        When an oversized non-target child CTE had to be materialized, these
        queries reference TEMP-table shards and **presuppose**
        :attr:`materialization_ddl` was executed first on the same session;
        :meth:`to_arrow` / :meth:`to_parquet` do that automatically.
        """
        return self._grouped().queries

    @property
    def materialization_ddl(self) -> List[str]:
        """The ``CREATE TEMP TABLE`` preamble (issue #7) that :attr:`query_groups`
        presupposes, or ``[]`` when no oversized child CTE needs materializing.

        Run these statements on the same connection/session before executing the
        grouped queries. :meth:`to_arrow` / :meth:`to_parquet` / :meth:`to_dataframe`
        run them for you.
        """
        mplan = self._grouped().materialization
        return list(mplan.ddl) if mplan is not None else []

    def to_dataframe(
        self, *, connection: Any = None, impute: bool = False, **impute_kwargs: Any
    ) -> pd.DataFrame:
        """Execute the query and return results as a DataFrame.

        Args:
            connection: An open psycopg connection to run on (required when the
                query references session ``TEMP`` tables — the integration
                harness). When ``None``, the fast single-query path uses
                ``records`` and the grouped/materialized path builds its own
                connection from the environment and closes it afterwards.
            impute: When True, run the opt-in imputation pass (count-like
                features → 0, measures left NULL unless ``measure_strategy`` is
                given, with ``<feature>__missing`` indicator columns). The
                default keeps the raw NULLs, since missingness is signal.
            **impute_kwargs: Forwarded to
                :func:`featurizer.imputation.impute_features`.

        A config that fits one valid query uses the fast (single-query) path; a
        wide or oversized-child config (issue #7) runs the column-group queries —
        and any TEMP-table materialization preamble — on one connection and
        re-joins them on ``(as_of_date, target_id)`` into the same indexed frame.
        Passing ``connection`` forces the one-connection path (so it can see
        session TEMP tables) regardless of width.

        Returns:
            DataFrame indexed by ['as_of_date', target_id]

        Raises:
            ValueError: If target entity doesn't define a primary ID
        """
        if self.target.id is None:
            raise ValueError(
                f"Target entity '{self.target.alias}' does not define a primary id."
            )
        grouped = self._grouped()
        if grouped.fits_single and connection is None:
            df = self._executor.to_dataframe(self.query, self.target.id.name)
        else:
            df = self._executor.to_dataframe_materialized(
                preamble_ddl=(
                    grouped.materialization.ddl
                    if grouped.materialization is not None
                    else []
                ),
                group_queries=grouped.queries,
                target_id=self.target.id.name,
                connection=connection,
            )
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
        """Execute the query and return Arrow output, no pandas hop.

        Streams the result out of PostgreSQL with binary ``COPY`` and decodes it
        column-by-column into Arrow, so SQL NULLs are preserved as Arrow nulls
        (never coerced to ``NaN``) and the full result set never materializes as
        a pandas frame. ``as_of_date`` and the target id are ordinary leading
        columns (no index), unlike :meth:`to_dataframe`.

        Sharding (issue #7): when the matrix fits in one query a single
        :class:`pyarrow.Table` is returned. When it is too wide for a single
        valid query (over PostgreSQL's 1664-entry target-list limit), an
        ``OrderedDict[str, pyarrow.Table]`` of column groups is returned instead
        — ``group_<NNN>`` -> table. Every group table leads with
        ``(as_of_date, <target id>)`` and the groups re-join on those keys to
        reconstruct the full matrix.

        Args:
            connection: An open psycopg connection to run ``COPY`` on. Required
                when the rendered query references session ``TEMP`` tables (the
                integration harness). When ``None``, a connection is built from
                ``DATABASE_URL`` / ``PG*`` and closed afterwards. A single
                connection is reused across all groups.
            numeric_as_float: Cast PostgreSQL ``numeric`` aggregate columns to
                ``float64`` (ML-ready, ``to_dataframe``-comparable). Set ``False``
                to keep exact ``decimal128``. NULLs are preserved either way.
            impute: When True, apply the Arrow-native imputation pass
                (:func:`featurizer.imputation.impute_arrow`) to each group:
                count-like features → 0, measures left null unless
                ``measure_strategy`` is given, with stable ``<feature>__missing``
                indicator columns. ``as_of_date`` and the target id are passed as
                ``key_columns`` and left untouched.
            **impute_kwargs: Forwarded to ``impute_arrow``. ``measure_strategy`` in
                ``{"mean","median"}`` additionally requires
                ``allow_full_matrix_fit=True`` (ADR-0001 leakage gate).

        Returns:
            A ``pyarrow.Table`` for a single-group config, otherwise an
            ``OrderedDict[str, pyarrow.Table]`` keyed by group id.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
            ValueError: If the target entity does not define a primary id, or a
                leaky measure strategy is requested without the opt-in.
        """
        if self.target.id is None:
            raise ValueError(
                f"Target entity '{self.target.alias}' does not define a primary id."
            )
        groups = self._arrow_groups(
            connection=connection,
            numeric_as_float=numeric_as_float,
            impute=impute,
            **impute_kwargs,
        )
        if len(groups) == 1:
            # Single-group config: preserve the original single-Table contract.
            return next(iter(groups.values()))
        return groups

    def to_parquet(
        self,
        path: str,
        *,
        connection: Any = None,
        numeric_as_float: bool = True,
        impute: bool = False,
        **impute_kwargs: Any,
    ) -> None:
        """Execute the query and write the result to Parquet.

        Thin wrapper over the Arrow path plus ``pyarrow.parquet.write_table``;
        all arguments (including the imputation contract and its ADR-0001 leakage
        gate) behave exactly as in :meth:`to_arrow`. NULLs are written as Parquet
        nulls.

        Sharding (issue #7): when the matrix fits in one query, a single Parquet
        file is written at ``path``. When it is too wide for one valid query,
        ``path`` is treated as a **directory** and one Parquet file per column
        group is written under it as ``group_<NNN>.parquet``. All group files
        re-join on ``(as_of_date, <target id>)`` to reconstruct the full matrix.

        Args:
            path: Destination ``.parquet`` file (single group) or output
                directory (multiple groups; created if absent).
            connection: See :meth:`to_arrow`.
            numeric_as_float: See :meth:`to_arrow`.
            impute: See :meth:`to_arrow`.
            **impute_kwargs: See :meth:`to_arrow`.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
        """
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        groups = self._arrow_groups(
            connection=connection,
            numeric_as_float=numeric_as_float,
            impute=impute,
            **impute_kwargs,
        )
        if len(groups) == 1:
            pq.write_table(next(iter(groups.values())), path)
            return

        import os as _os

        _os.makedirs(path, exist_ok=True)
        for gid, table in groups.items():
            pq.write_table(table, _os.path.join(path, f"{gid}.parquet"))
        logger.info(
            "Wrote {} column-group Parquet files to {} (re-join on {}).",
            len(groups),
            path,
            ("as_of_date", self.target.id.name) if self.target.id else "(as_of_date,)",
        )

    def to_tables(
        self,
        schema: str,
        *,
        connection: Any = None,
        table_prefix: str | None = None,
        create_schema: bool = True,
    ) -> List["FeatureGroupTable"]:
        """Persist the feature matrix as triage-style feature-group tables.

        Writes each column group as a persistent table
        ``"<schema>"."<stem>_group_<NNN>"`` keyed on ``(as_of_date, <target id>)``,
        the feature-group contract triage-pg consumes (issue #7). A config that
        fits one query writes a single ``<stem>_group_000``; a wide or
        oversized-child config writes one table per column group, all re-joinable
        on the keys. The issue-#7 intermediate shards stay ephemeral ``TEMP``
        tables — only the final groups are persisted.

        Idempotent: each target table is ``DROP TABLE IF EXISTS`` + ``CREATE TABLE
        … AS`` so a re-run replaces it cleanly.

        Alongside the group tables, the feature manifest is persisted as
        ``"<schema>"."<stem>_manifest"`` — one row per output column (label,
        lineage, generated description, and the ``feature_group`` it landed
        in), joinable to the group tables by column name. The returned list
        contains the feature-group tables only, as before.

        Args:
            schema: Destination schema (created if absent unless
                ``create_schema=False``).
            connection: An open psycopg connection to write on. When supplied the
                caller owns the transaction (nothing is committed here — the
                integration harness verifies within its rolled-back transaction);
                when ``None`` a connection is built from the environment, committed
                so the tables persist, and closed.
            table_prefix: Table-name stem; defaults to the target alias
                (``stores`` -> ``stores_group_000``).
            create_schema: Run ``CREATE SCHEMA IF NOT EXISTS`` first.

        Returns:
            The ordered manifest of created :class:`FeatureGroupTable`s.

        Raises:
            ValueError: If the target entity does not define a primary id.
        """
        if self.target.id is None:
            raise ValueError(
                f"Target entity '{self.target.alias}' does not define a primary id."
            )
        from .arrow import default_connection
        from .sharding import FeatureGroupTable

        grouped = self._grouped()
        preamble = (
            grouped.materialization.ddl if grouped.materialization is not None else []
        )
        stem = table_prefix or self.target.alias
        keys = list(grouped.key_columns)

        own_connection = connection is None
        conn = connection if connection is not None else default_connection()
        tables: List["FeatureGroupTable"] = []
        try:
            with conn.cursor() as cur:
                if create_schema:
                    cur.execute(f'create schema if not exists "{schema}"')
                for ddl in preamble:
                    cur.execute(ddl)
            analyze_as_of_dates(conn)  # planner-stats optimization (see executor)
            with conn.cursor() as cur:
                for gid, sql in grouped.queries.items():
                    name = f'"{schema}"."{stem}_{gid}"'
                    cur.execute(f"drop table if exists {name}")
                    cur.execute(f"create table {name} as\n{sql}")
                    tables.append(
                        FeatureGroupTable(name=name, group=gid, key_columns=list(keys))
                    )
                self._write_manifest_table(cur, schema, stem)
            if own_connection:
                conn.commit()
        finally:
            if own_connection:
                conn.close()
        logger.info(
            "Persisted {} feature-group table(s) + manifest to schema {!r} "
            "(re-join on {}).",
            len(tables),
            schema,
            tuple(keys),
        )
        return tables

    def _write_manifest_table(self, cur: Any, schema: str, stem: str) -> None:
        """Persist the feature manifest as ``"<schema>"."<stem>_manifest"``.

        One row per output feature column, including which feature-group table
        the column landed in (``feature_group``, joinable back to the
        ``<stem>_group_<NNN>`` tables by column name). Same contracts as the
        group tables: idempotent DROP+CREATE, and the caller owns the
        transaction. Values are inserted parameterized — labels and definitions
        contain quotes and arbitrary SQL text.
        """
        column_to_group: Dict[str, str] = {}
        for gid, names in self._sharder().column_groups().items():
            for column_name in names:
                column_to_group[column_name.replace('"', "")] = gid

        name = f'"{schema}"."{stem}_manifest"'
        cur.execute(f"drop table if exists {name}")
        cur.execute(f"""
            create table {name} (
                "column_name"   text not null,
                "label"         text not null,
                "truncated"     boolean not null,
                "kind"          text not null,
                "entity"        text,
                "source_alias"  text,
                "depth"         integer not null,
                "parents"       text[] not null,
                "interval"      text,
                "source_column" text,
                "value"         text,
                "description"   text not null,
                "definition"    text,
                "feature_group" text not null
            )
            """)
        rows = [
            (
                entry.column,
                entry.label,
                entry.truncated,
                entry.kind,
                entry.entity,
                entry.source_alias,
                entry.depth,
                entry.parents,
                entry.interval,
                entry.source_column,
                entry.value,
                entry.description,
                entry.definition,
                column_to_group.get(entry.column, "group_000"),
            )
            for entry in self.feature_manifest
        ]
        if rows:
            cur.executemany(
                f"insert into {name} values "
                "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                rows,
            )

    def _arrow_groups(
        self,
        *,
        connection: Any = None,
        numeric_as_float: bool = True,
        impute: bool = False,
        **impute_kwargs: Any,
    ) -> "OrderedDict[str, Any]":
        """Run every column group through the Arrow exporter on one connection.

        Always returns an ``OrderedDict`` (a single-group config yields one
        entry). Imputation, when requested, runs per group with the same
        ADR-0001 leakage gate as the single-query path; the gate is checked once
        up front so a leaky strategy is refused before any query runs.
        """
        from .arrow import ArrowExporter

        if impute:
            from .imputation import guard_full_matrix_fit

            guard_full_matrix_fit(
                impute_kwargs.get("measure_strategy", "none"),
                allow_full_matrix_fit=bool(impute_kwargs.get("allow_full_matrix_fit")),
                caller="to_arrow",
            )

        # Grouped queries + any TEMP-table materialization preamble (issue #7).
        # The single-group short-circuit + oversized warning live in _grouped().
        grouped = self._grouped()
        preamble = (
            grouped.materialization.ddl if grouped.materialization is not None else []
        )

        exporter = ArrowExporter()
        own_connection = connection is None
        conn = connection if connection is not None else exporter.open_connection()
        try:
            from collections import OrderedDict as _OrderedDict

            # Run the CREATE TEMP TABLE preamble first, on the *same* connection,
            # so the group queries' shard references resolve. The connection is
            # non-autocommit (default_connection / the harness), so ON COMMIT DROP
            # shards live for the whole transaction and drop when it closes.
            if preamble:
                with conn.cursor() as cur:
                    for ddl in preamble:
                        cur.execute(ddl)
            analyze_as_of_dates(conn)  # planner-stats optimization (see executor)

            tables: "OrderedDict[str, Any]" = _OrderedDict()
            for gid, sql in grouped.queries.items():
                table = exporter.to_arrow(
                    sql, connection=conn, numeric_as_float=numeric_as_float
                )
                if impute:
                    table = self._impute_group(table, **impute_kwargs)
                tables[gid] = table
            return tables
        finally:
            if own_connection:
                conn.close()

    def _impute_group(self, table: Any, **impute_kwargs: Any) -> Any:
        """Apply the Arrow imputation pass to one group's table.

        The leakage gate is already checked in :meth:`_arrow_groups`; strip the
        consumed ``allow_full_matrix_fit`` flag so ``impute_arrow`` does not see
        an unexpected keyword.
        """
        from .imputation import impute_arrow

        kwargs = dict(impute_kwargs)
        kwargs.pop("allow_full_matrix_fit", None)
        assert self.target.id is not None  # guarded by callers
        return impute_arrow(
            table,
            key_columns=("as_of_date", self.target.id.name),
            **kwargs,
        )

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
