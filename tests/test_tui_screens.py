"""The three featurizer screens, photographed over fake adapters. No database.

Every snapshot here is deterministic: a frozen clock, polling off, a fake
``PgSource`` that answers from memory, and the ``01-basic-aggregations``
config — the tree, the manifest and the SQL are whatever the engine renders
for it, so a planner change that moves a column shows up as a changed picture
rather than going unnoticed.

Skipped, not failed, where the ``tui`` extra cannot install (3.10, 3.11).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("lynkeus")

from lynkeus.app import ShellApp  # noqa: E402
from lynkeus.demo import DemoActions, DemoRuns, DemoStatus  # noqa: E402
from lynkeus.models import Health, QueryResult, TableDetail, TableInfo  # noqa: E402
from lynkeus.testing import press, settle  # noqa: E402

from featurizer import Featurizer  # noqa: E402
from featurizer.tui.adapters import ManifestRow, Materialization, Project  # noqa: E402
from featurizer.tui.screens import (  # noqa: E402
    ConfigScreen,
    ManifestScreen,
    SqlScreen,
    primitive_of,
)

NOW = datetime(2026, 9, 4, 20, 30)
CONFIG = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "01-basic-aggregations"
    / "config.yaml"
)
SIZE = (110, 34)


class FakeSource:
    """Answers the Data/Query/Explain calls from memory; never connects."""

    def __init__(self) -> None:
        self.explained: list[str] = []

    def health(self) -> Health:
        return Health(True, "pg 16")

    def query(self, statement: str, params: Any = None) -> QueryResult:
        return QueryResult(["as_of_date", "customer_id"], [["2024-01-01", 1]], 2.0)

    def explain(self, statement: str) -> QueryResult:
        self.explained.append(statement)
        return QueryResult(
            ["QUERY PLAN"],
            [
                ["Nested Loop  (cost=0.15..42.10 rows=100 width=840)"],
                ["  ->  Seq Scan on as_of_dates aod"],
                ["  ->  Subquery Scan on t"],
                ["Planning Time: 1.2 ms"],
                ["Execution Time: 8.4 ms"],
            ],
            9.0,
        )

    def tables(self) -> list[TableInfo]:
        return [
            TableInfo("example_01", "as_of_dates", "table", 12),
            TableInfo("example_01", "customers", "table", 100),
            TableInfo("example_01", "orders", "table", 1074),
        ]

    def table_detail(self, schema: str, name: str, sample: int = 3) -> TableDetail:
        return TableDetail(TableInfo(schema, name, "table", 100))


class FakeProject(Project):
    """A ``Project`` over the example config whose database answers are canned."""

    def __init__(self, materialized: bool = True) -> None:
        # ``Project.__init__`` loads the engine and stores the source; nothing
        # there touches the database.
        super().__init__(CONFIG, FakeSource(), featurizer=Featurizer(str(CONFIG)))  # type: ignore[arg-type]
        self.materialized = materialized

    def materializations(self) -> list[Materialization]:
        if not self.materialized:
            return []
        entries = self.featurizer.feature_manifest
        rows = [
            ManifestRow(e.column, e.label, "group_000", e.entity, e.kind, e.interval)
            for e in entries
        ]
        columns = ["as_of_date", "customer_id", "signup_date"] + [
            e.column for e in entries
        ]
        return [
            Materialization("example_01", "customers", rows, {"group_000": columns})
        ]

    def resolve_table(self, alias: str) -> Any:
        from featurizer.tui.adapters import TableFacts

        entity = self.featurizer.graph.entities[alias]
        return TableFacts(alias, entity.table, "example_01", entity.table, 100)


def cockpit(project: Project | None = None) -> ShellApp:
    """The shell with the demo's fake standard adapters and featurizer's screens."""
    project = project or FakeProject()
    return ShellApp(
        project="featurizer",
        subtitle="config.yaml · target customers",
        status_adapter=DemoStatus(),
        runs_adapter=DemoRuns(),
        actions_adapter=DemoActions(),
        source=project.source,
        project_screens=[
            ConfigScreen(project),
            ManifestScreen(project),
            SqlScreen(project),
        ],
        version="v1.2.0",
        poll_seconds=0,
        clock=lambda: NOW,
    )


# ----------------------------------------------------------------- pictures


def test_config_screen(shell_snapshot) -> None:
    """Tab 6: the entity graph, the planner facts, and the validator's verdict."""
    assert shell_snapshot(cockpit(), keys=["6", "v"], size=SIZE)


def test_manifest_screen(shell_snapshot) -> None:
    """Tab 7: every output column with entity, primitive, interval and group."""
    assert shell_snapshot(cockpit(), keys=["7"], size=SIZE)


def test_manifest_screen_filtered(shell_snapshot) -> None:
    """The filter is the library's glob: ``*amount|interval=P7D*`` narrows by label."""

    async def type_pattern(pilot) -> None:
        await pilot.press("slash")
        for ch in "*amount|interval=P7D*":
            await pilot.press(
                ch
                if ch not in "*|="
                else {"*": "asterisk", "|": "vertical_line", "=": "equals_sign"}[ch]
            )

    assert shell_snapshot(cockpit(), keys=["7"], before=type_pattern, size=SIZE)


def test_sql_screen(shell_snapshot) -> None:
    """Tab 8: one group, highlighted, and the plan after ``x``."""
    assert shell_snapshot(cockpit(), keys=["8", "x"], size=SIZE)


# ---------------------------------------------------------------- behaviour


def run(coro: Any) -> Any:
    """Drive a Pilot coroutine without pytest-asyncio (not a dev dependency)."""
    return asyncio.run(coro)


def test_the_filter_goes_through_manifest_matching() -> None:
    """The screen never globs on its own: what it shows is what the engine matches."""
    project = FakeProject()
    app = cockpit(project)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await press(pilot, "7")
            screen = app.screen_for("manifest")
            assert isinstance(screen, ManifestScreen)
            screen.pattern = "*orders.amount|interval=P30D*"
            screen.render_rows()
            await settle(pilot)
            expected = project.featurizer.columns_matching(
                "*orders.amount|interval=P30D*"
            )
            assert [e.column for e in screen.shown] == expected
            assert expected, "the example config must produce P30D amount features"

    run(drive())


def test_the_group_table_opens_on_query_only_when_materialized() -> None:
    project = FakeProject(materialized=False)
    app = cockpit(project)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await press(pilot, "7")
            screen = app.screen_for("manifest")
            assert isinstance(screen, ManifestScreen)
            assert screen.sql_for_selection() is None
            assert screen.groups, (
                "a single-query config assigns group_000 to every column"
            )
            assert set(screen.groups.values()) == {"group_000"}

    run(drive())

    materialized = FakeProject()
    app = cockpit(materialized)

    async def drive_materialized() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await press(pilot, "7")
            screen = app.screen_for("manifest")
            assert isinstance(screen, ManifestScreen)
            sql = screen.sql_for_selection()
            assert sql is not None
            assert '"example_01"."customers_group_000"' in sql
            assert '"as_of_date", "customer_id"' in sql

    run(drive_materialized())


def test_x_explains_through_the_data_source_and_never_runs_the_group() -> None:
    project = FakeProject()
    app = cockpit(project)
    source = project.source
    assert isinstance(source, FakeSource)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await press(pilot, "8", "x")
            screen = app.screen_for("sql")
            assert isinstance(screen, SqlScreen)
            assert screen.selected == "group_000"
            assert screen.sql_for_selection() is None, (
                "4 must not hand the group to Query"
            )

    run(drive())
    assert len(source.explained) == 1
    assert source.explained[0] == project.featurizer.query_groups["group_000"].strip()


def test_v_runs_the_validators_own_function() -> None:
    project = FakeProject()
    app = cockpit(project)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await press(pilot, "6", "v")
            screen = app.screen_for("config")
            assert isinstance(screen, ConfigScreen)
            assert screen.result is not None
            assert screen.result.is_valid

    run(drive())


def test_enter_on_an_entity_opens_its_table_on_data() -> None:
    project = FakeProject()
    app = cockpit(project)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await press(pilot, "6", "enter")
            assert app.current_slug == "data"
            data = app.screen_for("data")
            assert data.selected is not None  # type: ignore[attr-defined]
            assert data.selected.name == "customers"  # type: ignore[attr-defined]

    run(drive())


def test_primitive_of_reads_the_frozen_label_grammar() -> None:
    f = Featurizer(str(CONFIG))
    seen = {primitive_of(e) for e in f.feature_manifest}
    assert "one-hot" in seen, "customers.country is a fixed-vocabulary categorical"
    assert "direct" in seen
    assert {"COUNT", "SUM", "MEAN"} <= seen
