"""The cockpit's DB-free models and adapters: materializations, runs, pending work.

Skipped, not failed, where the ``tui`` extra cannot install (3.10, 3.11).
The verbs that need no lynkeus are in ``test_tui_cli.py`` and run everywhere.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("lynkeus")

from lynkeus.models import Health, RunState  # noqa: E402

from featurizer.cli import build_parser  # noqa: E402
from featurizer.tui.adapters import (  # noqa: E402
    FeaturizerActions,
    FeaturizerStatus,
    Inspection,
    ManifestRow,
    Materialization,
    Project,
    TableFacts,
    _like_literal,
    run_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "examples" / "01-basic-aggregations" / "config.yaml"

ROWS = [
    ManifestRow(
        f"c{i}", f"L{i}", "group_000" if i < 3 else "group_001", "t", "derived", None
    )
    for i in range(5)
]
COMPLETE = Materialization(
    "s",
    "t",
    ROWS,
    {
        "group_000": ["as_of_date", "t_id", "when", "c0", "c1", "c2"],
        "group_001": ["as_of_date", "t_id", "when", "c3", "c4", "extra"],
    },
)
BROKEN = Materialization(
    "s", "u", ROWS, {"group_000": ["as_of_date", "u_id", "c0", "c1"]}
)


def test_a_materialization_reads_its_groups_off_the_manifest_and_the_disk():
    assert COMPLETE.groups == ["group_000", "group_001"]
    assert COMPLETE.complete and COMPLETE.missing_groups == []
    assert COMPLETE.key_columns == ["as_of_date", "t_id"], "the frozen two-column lead"
    assert COMPLETE.carried_columns == ["when", "extra"]
    assert COMPLETE.stray_columns(["when"]) == ["extra"]
    assert COMPLETE.orphan_columns == []
    assert COMPLETE.present_columns("group_001") == ["c3", "c4"]
    assert COMPLETE.run_id == "s.t" and COMPLETE.manifest_table == "t_manifest"
    assert COMPLETE.table_name("group_001") == '"s"."t_group_001"'

    assert BROKEN.missing_groups == ["group_001"]
    assert not BROKEN.complete
    assert BROKEN.orphan_columns == ["c2", "c3", "c4"], "c2 is assigned but not on disk"
    assert BROKEN.present_columns("group_000") == ["c0", "c1"]
    assert BROKEN.present_columns("group_001") == []


def test_a_run_is_succeeded_only_when_every_group_table_exists():
    done = run_for(COMPLETE)
    assert done.state is RunState.SUCCEEDED
    assert done.started_at is None, "nobody wrote a timestamp; none is invented"
    assert done.run_id == "s.t" and done.name == "t"
    assert "5 features / 2 groups" in done.detail
    failed = run_for(BROKEN)
    assert failed.state is RunState.FAILED
    assert "1 missing" in failed.detail


def test_like_literal_escapes_the_underscore_every_stem_has():
    assert _like_literal("customers_") == "customers\\_"
    assert _like_literal("a%b\\c") == "a\\%b\\\\c"


class _NoSource:
    def health(self) -> Health:
        return Health(True, "pg 16")


def _project() -> Project:
    from featurizer import Featurizer

    return Project(CONFIG, _NoSource(), featurizer=Featurizer(str(CONFIG)))  # type: ignore[arg-type]


def _facts(alias: str, table: str, columns: list[str]) -> TableFacts:
    return TableFacts(alias, table, "example_01", table, 100, columns)


def test_pending_work_is_the_config_held_against_the_database():
    project = _project()
    status = FeaturizerStatus(project)
    f = project.featurizer
    good_customers = _facts(
        "customers", "customers", ["customer_id", "signup_date", "country", "age"]
    )
    good_orders = _facts(
        "orders",
        "orders",
        ["order_id", "customer_id", "order_date", "amount", "status"],
    )

    # Everything present, nothing materialized: one info line, no errors.
    items = status.pending(
        Inspection(
            Health(True), {"customers": good_customers, "orders": good_orders}, []
        )
    )
    assert [i.level for i in items] == ["info"]
    assert "materialize" in items[0].detail

    # A table the config names that does not exist, and one missing a column.
    items = status.pending(
        Inspection(
            Health(True),
            {
                "customers": TableFacts("customers", "customers"),
                "orders": _facts("orders", "orders", ["order_id", "customer_id"]),
            },
            [],
        )
    )
    errors = {i.name: i.detail for i in items if i.level == "error"}
    assert "does not exist" in errors["customers"]
    assert "order_date" in errors["orders"] and "amount" in errors["orders"]

    # A materialization whose manifest is one column behind the config.
    entries = f.feature_manifest
    rows = [
        ManifestRow(e.column, e.label, "group_000", e.entity, e.kind, e.interval)
        for e in entries[1:]
    ]
    columns = ["as_of_date", "customer_id", "signup_date"] + [r.column for r in rows]
    stale = Materialization("example_01", "customers", rows, {"group_000": columns})
    items = status.pending(
        Inspection(
            Health(True), {"customers": good_customers, "orders": good_orders}, [stale]
        )
    )
    drift = [i for i in items if "config drift" in i.detail]
    assert len(drift) == 1 and drift[0].level == "warn"
    assert "1 new in config" in drift[0].detail
    assert not [i for i in items if "does not describe" in i.detail], (
        "signup_date is the target's temporal index and rides along by design"
    )

    # A group table that is gone.
    gone = Materialization("example_01", "customers", rows, {})
    items = status.pending(
        Inspection(
            Health(True), {"customers": good_customers, "orders": good_orders}, [gone]
        )
    )
    assert any(i.level == "error" and "group_000" in i.detail for i in items)


def test_actions_name_the_cli_verbs_and_confirm_the_ones_that_replace_things():
    actions = FeaturizerActions(cwd=REPO_ROOT, parser=build_parser()).list()
    by_name = {a.name: a for a in actions}
    assert by_name["featurizer materialize"].destructive
    assert by_name["featurizer materialize"].args == "--config CONFIG --schema SCHEMA"
    assert by_name["featurizer render"].args == "--config CONFIG"
    assert not by_name["featurizer render"].destructive
    assert by_name["featurizer runs"].args == "list|show"
    if shutil.which("just"):
        assert by_name["just db-down"].destructive
        assert by_name["just bench-capture-golden"].destructive
        assert not by_name["just test-fast"].destructive
        assert "just pg_port" not in by_name, "a justfile variable is not a recipe"
