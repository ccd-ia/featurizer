"""The featurizer side of the lynkeus contract: three adapters over one config.

featurizer has no daemon and no run ledger. Its state is a config file plus a
database: the config declares the entity graph and the primitives; the
database holds the source tables and, after :meth:`Featurizer.to_tables`, the
``"<schema>"."<stem>_group_NNN"`` tables with ``"<stem>_manifest"`` beside
them. Everything here is a query over those two things — nothing is stored,
nothing is decided.

- :class:`Project` loads one :class:`~featurizer.Featurizer` and holds one
  :class:`~lynkeus.pg.PgSource`; :meth:`Project.inspect` is the single
  read-only pass the Status screen renders.
- :class:`FeaturizerStatus` turns an inspection into a lynkeus ``Status``.
- :class:`FeaturizerRuns` treats every materialization found in the database
  as a run: one ``Run`` per ``<stem>_manifest`` table. There is nothing to
  stream and no start time, because nobody wrote one.
- :class:`FeaturizerActions` runs ``just`` recipes and ``python -m featurizer``
  verbs as subprocesses, the only way the cockpit starts work.

Written in Python 3.10 syntax on purpose: ``basedpyright`` checks this package
with the library's floor even though lynkeus only installs on 3.12+.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from lynkeus.actions import SubprocessActions, argparse_actions, just_actions
from lynkeus.models import (
    Action,
    Gauge,
    Health,
    PendingItem,
    Run,
    RunDetail,
    RunEvent,
    RunState,
    Series,
    Stage,
    Status,
)
from lynkeus.pg import PgSource
from psycopg import sql

from ..featurizer import Featurizer
from ..primitives import Entity, Variable
from ..validation import ValidationResult, validate_config

#: Columns a ``<stem>_manifest`` table must carry to count as one. The full
#: shape is written by ``Featurizer._write_manifest_table``; these five are the
#: ones the cockpit reads, so a table merely *named* ``*_manifest`` is not
#: mistaken for a materialization.
MANIFEST_REQUIRED: tuple[str, ...] = (
    "column_name",
    "label",
    "feature_group",
    "kind",
    "entity",
)

_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")

_MANIFESTS_SQL = """
select n.nspname as schema, c.relname as name
from   pg_class c
join   pg_namespace n on n.oid = c.relnamespace
where  c.relkind in ('r', 'p')
  and  c.relname like %(pattern)s escape '\\'
  and  n.nspname <> all(%(system)s::text[])
  and  left(n.nspname, 7) <> 'pg_temp'
  and  (%(schema)s::text is null or n.nspname = %(schema)s)
  and  (select array_agg(a.attname::text)
        from   pg_attribute a
        where  a.attrelid = c.oid and a.attnum > 0 and not a.attisdropped)
       @> %(required)s::text[]
order by n.nspname, c.relname
"""

_GROUP_TABLES_SQL = """
select c.relname as name,
       array_agg(a.attname::text order by a.attnum) as columns
from   pg_class c
join   pg_namespace n on n.oid = c.relnamespace
join   pg_attribute a on a.attrelid = c.oid and a.attnum > 0 and not a.attisdropped
where  n.nspname = %(schema)s
  and  c.relkind in ('r', 'p')
  and  c.relname like %(pattern)s escape '\\'
group by c.relname
order by c.relname
"""

_SOURCE_TABLE_SQL = """
select n.nspname as schema, c.relname as name,
       case when c.reltuples < 0 then null else c.reltuples::bigint end as rows_estimate,
       (select array_agg(a.attname::text order by a.attnum)
        from   pg_attribute a
        where  a.attrelid = c.oid and a.attnum > 0 and not a.attisdropped) as columns
from   pg_class c
join   pg_namespace n on n.oid = c.relnamespace
where  c.oid = to_regclass(%(name)s)
"""


def _like_literal(text: str) -> str:
    """Escape ``text`` so ``LIKE … ESCAPE '\\'`` matches it literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ---------------------------------------------------------------- models


@dataclass(frozen=True)
class ManifestRow:
    """One row of a persisted ``<stem>_manifest`` table, as the cockpit reads it."""

    column: str
    label: str
    group: str
    entity: Optional[str]
    kind: str
    interval: Optional[str]


@dataclass(frozen=True)
class Materialization:
    """What one ``to_tables()`` call left in the database.

    ``tables`` maps a group id (``group_000``) to the columns of that group
    table as they are on disk; a group the manifest names but that has no
    table is simply absent from it.
    """

    schema: str
    stem: str
    manifest: List[ManifestRow]
    tables: Dict[str, List[str]]

    @property
    def run_id(self) -> str:
        """``schema.stem`` — the only identity a materialization has."""
        return f"{self.schema}.{self.stem}"

    @property
    def manifest_table(self) -> str:
        return f"{self.stem}_manifest"

    @property
    def groups(self) -> List[str]:
        """Every group the manifest names, in id order."""
        return sorted({row.group for row in self.manifest})

    @property
    def missing_groups(self) -> List[str]:
        """Groups the manifest names that have no table on disk."""
        return [gid for gid in self.groups if gid not in self.tables]

    @property
    def complete(self) -> bool:
        return not self.missing_groups

    def columns_for(self, group: str) -> List[str]:
        """Manifest columns assigned to ``group``."""
        return [row.column for row in self.manifest if row.group == group]

    @property
    def manifest_columns(self) -> List[str]:
        return [row.column for row in self.manifest]

    @property
    def key_columns(self) -> List[str]:
        """The two leading columns every group table shares: as_of_date + the id.

        ADR-0015 freezes that lead (``GroupedQueries.key_columns``), so the
        first two columns of any group table present are the keys.
        """
        for gid in self.groups:
            if gid in self.tables:
                return self.tables[gid][:2]
        return []

    @property
    def carried_columns(self) -> List[str]:
        """Group-table columns that are neither keys nor manifest columns.

        The target's own index columns (its temporal index, a spatial column)
        ride along in the group tables without being features, so they are
        absent from the manifest by construction. Anything else here is a
        column the manifest genuinely does not describe.
        """
        described = set(self.manifest_columns) | set(self.key_columns)
        seen: List[str] = []
        for columns in self.tables.values():
            for c in columns:
                if c not in described and c not in seen:
                    seen.append(c)
        return seen

    def present_columns(self, group: str) -> List[str]:
        """Manifest columns of ``group`` that its table actually has."""
        on_disk = set(self.tables.get(group, []))
        return [c for c in self.columns_for(group) if c in on_disk]

    @property
    def orphan_columns(self) -> List[str]:
        """Manifest columns found on no group table."""
        on_disk = {c for columns in self.tables.values() for c in columns}
        return [c for c in self.manifest_columns if c not in on_disk]

    def stray_columns(self, expected: Sequence[str] = ()) -> List[str]:
        """Carried columns the config does not account for either.

        ``expected`` is what the config knows rides along (the target's index
        columns); with the config in hand the Status screen passes them, so a
        temporal index is not reported as drift on every materialization.
        """
        return [c for c in self.carried_columns if c not in expected]

    def table_name(self, group: str) -> str:
        """The quoted, schema-qualified name of one group table."""
        return f'"{self.schema}"."{self.stem}_{group}"'


@dataclass(frozen=True)
class TableFacts:
    """One entity's source table as the database has it (or does not)."""

    alias: str
    table: str
    """The name as the config wrote it (bare or schema-qualified)."""
    schema: Optional[str] = None
    name: Optional[str] = None
    rows_estimate: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def exists(self) -> bool:
        return self.name is not None


@dataclass(frozen=True)
class Inspection:
    """Everything the Status screen shows, gathered in one read-only pass."""

    health: Health
    tables: Dict[str, TableFacts]
    materializations: List[Materialization]


# ---------------------------------------------------------------- queries


def find_materializations(
    conn: psycopg.Connection[Any], schema: Optional[str] = None
) -> List[Materialization]:
    """Every ``<stem>_manifest`` table with its group tables, on ``conn``.

    A materialization is recognised by its manifest, never by a group table
    alone: the manifest is what says which groups *should* exist.
    """
    found: List[Materialization] = []
    rows = conn.execute(
        _MANIFESTS_SQL,
        {
            "pattern": "%\\_manifest",
            "system": list(_SYSTEM_SCHEMAS),
            "schema": schema,
            "required": list(MANIFEST_REQUIRED),
        },
    ).fetchall()
    for row in rows:
        found.append(_read_materialization(conn, str(row["schema"]), str(row["name"])))
    return found


def _read_materialization(
    conn: psycopg.Connection[Any], schema: str, manifest_table: str
) -> Materialization:
    stem = manifest_table[: -len("_manifest")]
    manifest = [
        ManifestRow(
            column=str(r["column_name"]),
            label=str(r["label"]),
            group=str(r["feature_group"]),
            entity=r["entity"],
            kind=str(r["kind"]),
            interval=r["interval"],
        )
        for r in conn.execute(
            sql.SQL(
                'select "column_name", "label", "feature_group", "entity", '
                '"kind", "interval" from {}'
            ).format(sql.Identifier(schema, manifest_table))
        ).fetchall()
    ]
    tables: Dict[str, List[str]] = {}
    prefix = f"{stem}_"
    for r in conn.execute(
        _GROUP_TABLES_SQL,
        {"schema": schema, "pattern": _like_literal(prefix) + "group\\_%"},
    ).fetchall():
        name = str(r["name"])
        tables[name[len(prefix) :]] = [str(c) for c in r["columns"]]
    return Materialization(schema, stem, manifest, tables)


def _source_table(conn: psycopg.Connection[Any], entity: Entity) -> TableFacts:
    """Resolve one entity's table through ``to_regclass``, inside a savepoint.

    A table name the config spells in a way PostgreSQL cannot parse raises
    rather than returning null; the savepoint keeps that from aborting the
    rest of the inspection, and the message lands in ``error``.
    """
    conn.execute("savepoint source_table")
    try:
        row = conn.execute(_SOURCE_TABLE_SQL, {"name": entity.table}).fetchone()
    except psycopg.Error as exc:
        conn.execute("rollback to savepoint source_table")
        return TableFacts(entity.alias, entity.table, error=str(exc).splitlines()[0])
    conn.execute("release savepoint source_table")
    if row is None:
        return TableFacts(entity.alias, entity.table)
    return TableFacts(
        entity.alias,
        entity.table,
        schema=str(row["schema"]),
        name=str(row["name"]),
        rows_estimate=row["rows_estimate"],
        columns=[str(c) for c in (row["columns"] or [])],
    )


# ---------------------------------------------------------------- project


class Project:
    """One config against one database.

    Args:
        config_path: The YAML the cockpit shows.
        source: Where to read; ``PgSource.from_env()`` in normal use.
        schema: Narrow the materialization scan to one schema. ``None`` scans
            every user schema, which is right for a throwaway database and
            noisy for a shared one.
        featurizer: A pre-loaded engine, for callers that already have one.
    """

    def __init__(
        self,
        config_path: str | Path,
        source: PgSource,
        *,
        schema: Optional[str] = None,
        featurizer: Optional[Featurizer] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.source = source
        self.schema = schema
        self.featurizer = featurizer or Featurizer(str(self.config_path))

    # ------------------------------------------------------------ config
    @property
    def target(self) -> Entity:
        return self.featurizer.target

    def wanted_columns(self, entity: Entity) -> List[str]:
        """Every column the config names on ``entity``: indexes, keys, variables."""
        names: List[str] = [ix.name for ix in entity.indexes]
        names += [key.name for key in entity.keys]
        names += [f.name for f in entity.features if isinstance(f, Variable)]
        seen: List[str] = []
        for name in names:
            bare = name.replace('"', "")
            if bare not in seen:
                seen.append(bare)
        return seen

    def validate(self) -> ValidationResult:
        """The ``featurizer validate`` verb's own check, unprinted."""
        return validate_config(str(self.config_path))

    # ----------------------------------------------------------- queries
    def inspect(self) -> Inspection:
        """One read-only transaction: health, every source table, every manifest."""
        health = self.source.health()
        if not health.connected:
            return Inspection(health, {}, [])
        with self.source.connect() as conn:
            conn.execute("set transaction read only")
            tables = {
                entity.alias: _source_table(conn, entity)
                for entity in self.featurizer.entities
            }
            materializations = find_materializations(conn, self.schema)
            conn.rollback()
        return Inspection(health, tables, materializations)

    def materializations(self) -> List[Materialization]:
        """Every materialization in scope, without the source-table pass."""
        with self.source.connect() as conn:
            conn.execute("set transaction read only")
            found = find_materializations(conn, self.schema)
            conn.rollback()
        return found

    def resolve_table(self, alias: str) -> Optional[TableFacts]:
        """Where one entity's table lives, or ``None`` when it does not exist."""
        entity = self.featurizer.graph.entities.get(alias)
        if entity is None:
            return None
        with self.source.connect() as conn:
            conn.execute("set transaction read only")
            facts = _source_table(conn, entity)
            conn.rollback()
        return facts if facts.exists else None


# --------------------------------------------------------------- adapters


def run_for(materialization: Materialization) -> Run:
    """A materialization as a lynkeus ``Run``: done when every group table exists."""
    n_groups = len(materialization.groups)
    return Run(
        run_id=materialization.run_id,
        name=materialization.stem,
        state=RunState.SUCCEEDED if materialization.complete else RunState.FAILED,
        started_at=None,
        finished_at=None,
        detail=(
            f"{len(materialization.manifest)} features / {n_groups} "
            f"group{'s' if n_groups != 1 else ''}"
            + (
                f" · {len(materialization.missing_groups)} missing"
                if not materialization.complete
                else ""
            )
        ),
    )


class FeaturizerStatus:
    """``StatusAdapter``: the config's facts and the database's answers."""

    def __init__(self, project: Project) -> None:
        self.project = project

    def status(self) -> Status:
        """Facts from the config, gauges and pending work from queries."""
        f = self.project.featurizer
        inspection = self.project.inspect()
        manifest = f.feature_manifest
        groups = f.query_groups
        as_of = sum(1 for r in f.relationships if r.temporal_mode)
        entities = list(f.entities)
        extra = {
            "config": str(self.project.config_path),
            "target": f.target.alias,
            "entities": f"{len(entities)} · " + ", ".join(e.alias for e in entities),
            "relationships": f"{len(f.relationships)}"
            + (f" · {as_of} as-of" if as_of else ""),
            "primitives": (
                f"{len(f.aggregations)} aggregations · "
                f"{len(f.transformations)} transformations"
            ),
            "temporal": (
                f"intervals {', '.join(f.intervals) if f.intervals else 'none'}"
                f" · max_depth {f.max_depth} · boundary {f.as_of_boundary}"
            ),
            "manifest": f"{len(manifest)} columns · {len(groups)} group"
            + ("s" if len(groups) != 1 else ""),
        }
        gauges: List[Gauge] = []
        for entity in entities:
            facts = inspection.tables.get(entity.alias)
            if facts is None or not facts.exists:
                gauges.append(Gauge(f"{entity.alias} rows", 0, note="table missing"))
                continue
            gauges.append(
                Gauge(
                    f"{entity.alias} rows",
                    float(facts.rows_estimate or 0),
                    note=f"{facts.schema}.{facts.name}"
                    + (" · never analyzed" if facts.rows_estimate is None else ""),
                )
            )
        for m in inspection.materializations:
            gauges.append(
                Gauge(
                    m.run_id,
                    float(len(m.groups) - len(m.missing_groups)),
                    total=float(len(m.groups)),
                    note="group tables",
                )
            )
        return Status(
            project="featurizer",
            database=inspection.health,
            last_runs=[run_for(m) for m in inspection.materializations][:5],
            pending=self.pending(inspection),
            extra=extra,
            gauges=gauges,
            series=[
                Series(
                    "materializations over time",
                    [],
                    "no run ledger: a materialization is a table, not an event",
                )
            ],
        )

    def pending(self, inspection: Inspection) -> List[PendingItem]:
        """Every line is a comparison between the config and the database."""
        items: List[PendingItem] = []
        if not inspection.health.connected:
            return [PendingItem("database", inspection.health.detail, level="error")]
        f = self.project.featurizer
        for entity in f.entities:
            facts = inspection.tables.get(entity.alias)
            if facts is None:
                continue
            if facts.error:
                items.append(
                    PendingItem(
                        entity.alias, f"{facts.table}: {facts.error}", level="error"
                    )
                )
                continue
            if not facts.exists:
                items.append(
                    PendingItem(
                        entity.alias,
                        f"table {facts.table} does not exist",
                        level="error",
                    )
                )
                continue
            missing = [
                c for c in self.project.wanted_columns(entity) if c not in facts.columns
            ]
            if missing:
                items.append(
                    PendingItem(
                        entity.alias,
                        f"{facts.table} lacks {len(missing)} column"
                        f"{'s' if len(missing) != 1 else ''}: {', '.join(missing)}",
                        level="error",
                    )
                )
        if not inspection.materializations:
            scope = f"in {self.project.schema}" if self.project.schema else "anywhere"
            items.append(
                PendingItem(
                    "materializations",
                    f"none found {scope} — `featurizer materialize` writes one",
                    level="info",
                )
            )
        config_labels = {entry.label for entry in f.feature_manifest}
        for m in inspection.materializations:
            if m.missing_groups:
                items.append(
                    PendingItem(
                        m.run_id,
                        f"{len(m.missing_groups)} group table"
                        f"{'s' if len(m.missing_groups) != 1 else ''} missing: "
                        + ", ".join(m.missing_groups),
                        level="error",
                    )
                )
            orphans = m.orphan_columns
            if orphans:
                items.append(
                    PendingItem(
                        m.run_id,
                        f"{len(orphans)} manifest column"
                        f"{'s' if len(orphans) != 1 else ''} on no group table",
                        level="warn",
                    )
                )
            strays = m.stray_columns([ix.name for ix in f.target.indexes])
            if strays:
                items.append(
                    PendingItem(
                        m.run_id,
                        f"{len(strays)} column{'s' if len(strays) != 1 else ''} on "
                        "disk the manifest does not describe",
                        level="warn",
                    )
                )
            persisted = {row.label for row in m.manifest}
            added = config_labels - persisted
            gone = persisted - config_labels
            if added or gone:
                items.append(
                    PendingItem(
                        m.run_id,
                        f"config drift: {len(added)} new in config · {len(gone)} no "
                        "longer produced — re-run materialize",
                        level="warn",
                    )
                )
        return items


class FeaturizerRuns:
    """``RunsAdapter`` over the materializations in the database.

    There is no ledger and this adapter does not invent one: a run is a
    ``<stem>_manifest`` table, its stages are the group tables the manifest
    names, and its state is whether they all exist. ``started_at`` is ``None``
    because nobody wrote a timestamp, and ``events`` yields nothing because
    there is nothing to stream.
    """

    #: Shown in the Runs screen's progress panel title.
    mode = "nothing to stream"

    def __init__(self, source: PgSource, schema: Optional[str] = None) -> None:
        self.source = source
        self.schema = schema

    def _scan(self) -> List[Materialization]:
        with self.source.connect() as conn:
            conn.execute("set transaction read only")
            found = find_materializations(conn, self.schema)
            conn.rollback()
        return found

    def _find(self, run_id: str) -> Materialization:
        found = self._scan()
        exact = [m for m in found if m.run_id == run_id]
        if exact:
            return exact[0]
        by_prefix = [m for m in found if m.run_id.startswith(run_id)]
        if len(by_prefix) == 1:
            return by_prefix[0]
        known = ", ".join(m.run_id for m in found) or "none"
        raise KeyError(
            f"no materialization {run_id!r}"
            + (" (ambiguous prefix)" if by_prefix else "")
            + f" — known: {known}"
        )

    def list(self, limit: int = 50) -> List[Run]:
        """One run per manifest table, in schema.stem order."""
        return [run_for(m) for m in self._scan()][:limit]

    def show(self, run_id: str) -> RunDetail:
        """One stage per group the manifest names: columns present of assigned."""
        m = self._find(run_id)
        stages = [
            Stage(
                gid,
                len(m.present_columns(gid)),
                len(m.columns_for(gid)),
                "" if gid in m.tables else "table missing",
            )
            for gid in m.groups
        ]
        return RunDetail(
            run_for(m),
            stages,
            {
                "schema": m.schema,
                "stem": m.stem,
                "keys": ", ".join(m.key_columns) or "—",
                "carried": ", ".join(m.carried_columns) or "—",
                "manifest": m.manifest_table,
            },
        )

    def events(self, run_id: str) -> Iterator[RunEvent | None]:
        """Nothing to stream: a table has no progress log."""
        del run_id
        return iter(())

    def cancel(self, run_id: str) -> None:
        """A materialization is a table, not a process."""
        raise RuntimeError(
            f"{run_id} is a set of tables, not a process — drop them with SQL, "
            "or re-run `featurizer materialize` to replace them"
        )


class FeaturizerActions(SubprocessActions):
    """``ActionsAdapter``: the justfile's recipes and the CLI's verbs.

    There is no console script, so the ``module`` fallback (``python -m
    featurizer``) is the normal path, not the exception.
    """

    prefix = "featurizer"
    module = "featurizer"
    #: Confirmed whatever their wording: ``db-down`` removes the container,
    #: ``bench-capture-golden`` is run-once by the justfile's own comment, and
    #: ``materialize`` drops and recreates the group tables it writes.
    destructive = frozenset(
        {"just db-down", "just bench-capture-golden", "featurizer materialize"}
    )

    def __init__(self, cwd: Optional[Path] = None, parser: Any = None) -> None:
        super().__init__(cwd)
        self.parser = parser

    def list(self) -> List[Action]:
        """Recipes first, then every verb of the CLI's parser."""
        from ..cli import build_parser

        parser = self.parser if self.parser is not None else build_parser()
        return just_actions(self.cwd, self.is_destructive) + argparse_actions(
            parser, self.prefix, self.is_destructive
        )

    def run(self, name: str, args: Sequence[str]) -> subprocess.Popen[str]:
        """Start a recipe or verb (see ``SubprocessActions.run``)."""
        return super().run(name, list(args))


def source_for() -> PgSource:
    """The database, from ``PG*`` / ``DATABASE_URL`` only — never a guess."""
    return PgSource.from_env()


#: Queries the Query screen offers before the user writes one. The manifest
#: query is the cockpit's own scan; the group-table one lists what to_tables()
#: wrote, with sizes, so a database can be tidied by eye.
SAVED_QUERIES: Dict[str, str] = {
    "manifests": (
        "-- every materialization: one <stem>_manifest table each\n"
        "select n.nspname as schema, c.relname as manifest,\n"
        "       pg_size_pretty(pg_total_relation_size(c.oid)) as size\n"
        "from   pg_class c\n"
        "join   pg_namespace n on n.oid = c.relnamespace\n"
        "where  c.relkind = 'r'\n"
        "  and  c.relname like '%\\_manifest' escape '\\'\n"
        "  and  n.nspname not in ('pg_catalog', 'information_schema', 'pg_toast')\n"
        "order by 1, 2;"
    ),
    "group tables": (
        "-- feature-group tables, with their sizes\n"
        'select n.nspname as schema, c.relname as "table",\n'
        "       case when c.reltuples < 0 then null else c.reltuples::bigint end"
        " as rows_estimate,\n"
        "       pg_size_pretty(pg_total_relation_size(c.oid)) as size\n"
        "from   pg_class c\n"
        "join   pg_namespace n on n.oid = c.relnamespace\n"
        "where  c.relkind = 'r'\n"
        "  and  c.relname ~ '_group_[0-9]+$'\n"
        "  and  n.nspname not in ('pg_catalog', 'information_schema', 'pg_toast')\n"
        "order by 1, 2;"
    ),
}
