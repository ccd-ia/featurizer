"""``featurizer tui``: the lynkeus shell with featurizer's adapters and screens.

Credentials are resolved exactly as the integration tests resolve them: from
``PG*`` / ``DATABASE_URL`` via ``PgSource.from_env()``. Nothing in this module
names a host, and nothing here runs featurizer code that writes — the only
writer is the ``materialize`` verb, started from Actions as a subprocess.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional

from lynkeus.app import ShellApp
from lynkeus.pg import PgSource

from .adapters import (
    SAVED_QUERIES,
    FeaturizerActions,
    FeaturizerRuns,
    FeaturizerStatus,
    Project,
    source_for,
)

HELP_EXTRA = (
    "[$primary]6[/] Config — the entity graph as a tree: the target at the "
    "root, each relationship with its keys and temporal mode, the primitives "
    "and intervals. [$primary]v[/] validates the file the way "
    "[$primary]featurizer validate[/] does and shows the findings inline; "
    "[$primary]enter[/] on an entity opens its source table on Data.\n"
    "[$primary]7[/] Manifest — every output column with its entity, primitive, "
    "interval and group. [$primary]/[/] filters through the same glob "
    "[$primary]manifest_matching[/] uses (labels, never physical names); "
    "[$primary]y[/] copies the matching column list; [$primary]4[/] opens the "
    "group table on Query.\n"
    "[$primary]8[/] SQL — the live preview: one column group at a time, "
    "read-only. [$primary]x[/] explains it (analyze, rolled back); "
    "[$primary]y[/] copies it. Running it is the "
    "[$primary]featurizer materialize[/] action, confirmed like any other.\n"
    "featurizer keeps no run ledger: a run here is a materialization found "
    "in the database, one per [$primary]<stem>_manifest[/] table."
)


def featurizer_version() -> str:
    """``vX.Y.Z`` from the installed distribution, or empty when not installed."""
    try:
        return f"v{version('featurizer')}"
    except PackageNotFoundError:  # pragma: no cover — a checkout never synced
        return ""


def state_dir(config_path: Path) -> Path:
    """Where the Query screen keeps saved queries and CSV exports, per config."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "featurizer" / "tui" / config_path.stem


def saved_queries(project: Project) -> Dict[str, str]:
    """The static queries, plus one over this config's own manifest when it exists."""
    queries = dict(SAVED_QUERIES)
    try:
        found = project.materializations()
    except Exception:  # noqa: BLE001 — the Status screen reports the outage
        return queries
    stem = project.target.alias
    mine = [m for m in found if m.stem == stem]
    for m in mine[:1]:
        queries[f"{m.stem} manifest"] = (
            f"-- the manifest to_tables() wrote for {m.run_id}\n"
            'select "column_name", "label", "feature_group", "entity", "kind",\n'
            '       "interval", "truncated"\n'
            f'from   "{m.schema}"."{m.manifest_table}"\n'
            'order by "feature_group", "column_name";'
        )
    return queries


def build_app(
    config: str | Path,
    *,
    schema: Optional[str] = None,
    poll_seconds: float = 5.0,
    cwd: Optional[Path] = None,
    clock: Optional[Callable[[], datetime]] = None,
    parser: Any = None,
    source: Optional[PgSource] = None,
    with_project_screens: bool = True,
) -> ShellApp:
    """Assemble the shell over one config and this environment's database."""
    config_path = Path(config)
    project = Project(config_path, source or source_for(), schema=schema)
    actions = FeaturizerActions(cwd, parser)
    screens: list[Any] = []
    if with_project_screens:
        from .screens import project_screens

        screens = project_screens(project)
    return ShellApp(
        project="featurizer",
        subtitle=f"{config_path.name} · target {project.target.alias}",
        status_adapter=FeaturizerStatus(project),
        runs_adapter=FeaturizerRuns(project.source, schema),
        actions_adapter=actions,
        source=project.source,
        project_screens=screens,
        saved_queries=saved_queries(project),
        version=featurizer_version(),
        poll_seconds=poll_seconds,
        clock=clock,
        state_dir=state_dir(config_path),
        help_extra=HELP_EXTRA,
    )
