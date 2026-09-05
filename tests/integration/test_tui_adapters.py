"""The cockpit's adapters against a real PostgreSQL, with one config materialized.

The example ``01-basic-aggregations`` config is run against a schema this
module creates and drops: seed the three tables, read the status before and
after ``to_tables()``, then break things one at a time — a group table gone, a
source table gone, a manifest row behind the config — and check that each
shows up as pending work rather than as a crash.

Needs the ``tui`` extra (Python 3.12+) and a configured database; skips
without either.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("lynkeus")

import psycopg  # noqa: E402
from lynkeus.models import RunState  # noqa: E402
from lynkeus.pg import PgSource  # noqa: E402
from psycopg import sql  # noqa: E402

from featurizer import Featurizer  # noqa: E402
from featurizer.tui.adapters import (  # noqa: E402
    FeaturizerRuns,
    FeaturizerStatus,
    Project,
)

pytestmark = pytest.mark.integration

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "01-basic-aggregations"
    / "config.yaml"
)


@pytest.fixture
def cockpit_schema(pg_conn, monkeypatch) -> Iterator[str]:
    """A committed schema holding the example's three tables, on the search_path.

    ``pg_conn`` decides whether a database is configured (and skips otherwise);
    the work here needs committed tables that other connections can see, so
    it opens its own autocommit connection to the same server. ``PGOPTIONS``
    puts the schema first on every new connection's ``search_path`` — the
    cockpit's, and ``to_tables()``'s — the way the examples pin theirs.
    """
    name = f"tui_t_{uuid.uuid4().hex[:8]}"
    schema = sql.Identifier(name)
    conn = psycopg.connect(
        pg_conn.info.dsn, password=pg_conn.info.password, autocommit=True
    )
    monkeypatch.setenv("PGOPTIONS", f"-csearch_path={name}")
    try:
        conn.execute(sql.SQL("create schema {}").format(schema))
        conn.execute(sql.SQL("set search_path to {}").format(schema))
        conn.execute(
            "create table customers (customer_id integer primary key, "
            "signup_date date not null, country text not null, age integer not null)"
        )
        conn.execute(
            "create table orders (order_id integer primary key, "
            "customer_id integer not null references customers, "
            "order_date date not null, amount double precision not null, "
            "status text not null)"
        )
        conn.execute("create table as_of_dates (as_of_date date primary key)")
        conn.execute(
            "insert into customers values (1, '2023-01-05', 'US', 30), "
            "(2, '2023-02-10', 'UK', 41), (3, '2023-03-15', 'DE', 27)"
        )
        conn.execute(
            "insert into orders values "
            "(1, 1, '2023-06-01', 10.0, 'completed'), "
            "(2, 1, '2023-06-20', 25.5, 'completed'), "
            "(3, 2, '2023-07-01', 7.25, 'pending'), "
            "(4, 3, '2023-12-24', 99.0, 'cancelled')"
        )
        conn.execute("insert into as_of_dates values ('2024-01-01'), ('2024-02-01')")
        conn.execute("analyze customers")
        conn.execute("analyze orders")
        yield name
    finally:
        conn.execute(sql.SQL("drop schema {} cascade").format(schema))
        conn.close()


def test_status_and_runs_before_and_after_materializing(cockpit_schema: str) -> None:
    name = cockpit_schema
    project = Project(CONFIG, PgSource.from_env(), schema=name)
    status = FeaturizerStatus(project)
    runs = FeaturizerRuns(project.source, name)

    # Before: tables present, nothing materialized.
    before = status.status()
    assert before.database.connected
    gauges = {g.name: g for g in before.gauges}
    assert gauges["customers rows"].value == 3
    assert gauges["orders rows"].value == 4
    assert gauges["customers rows"].note == f"{name}.customers"
    assert [i.level for i in before.pending] == ["info"]
    assert before.last_runs == []
    assert before.series[0].empty and "no run ledger" in before.series[0].empty_note
    assert runs.list() == []

    # Materialize through the frozen method, exactly as the verb does.
    tables = Featurizer(str(CONFIG)).to_tables(name)
    assert [t.group for t in tables] == ["group_000"]

    after = status.status()
    assert after.pending == [], [i.detail for i in after.pending]
    assert [r.run_id for r in after.last_runs] == [f"{name}.customers"]
    assert after.last_runs[0].state is RunState.SUCCEEDED
    assert after.last_runs[0].started_at is None
    gauge = {g.name: g for g in after.gauges}[f"{name}.customers"]
    assert (gauge.value, gauge.total) == (1.0, 1.0)

    # Runs: one per manifest table, one stage per group, keys from the lead.
    listed = runs.list()
    assert [r.run_id for r in listed] == [f"{name}.customers"]
    detail = runs.show(f"{name}.customers")
    manifest_width = len(project.featurizer.feature_manifest)
    assert [(s.name, s.done, s.total) for s in detail.stages] == [
        ("group_000", manifest_width, manifest_width)
    ]
    assert detail.meta["keys"] == "as_of_date, customer_id"
    assert detail.meta["carried"] == "signup_date"
    assert runs.show(name[:5]).run.run_id == f"{name}.customers", "unique prefix"
    assert list(runs.events(f"{name}.customers")) == []
    with pytest.raises(RuntimeError):
        runs.cancel(f"{name}.customers")
    with pytest.raises(KeyError):
        runs.show("nope")

    # The Manifest screen's group lookup and the Config screen's table resolve.
    found = project.materializations()
    assert len(found) == 1 and found[0].complete
    assert {row.group for row in found[0].manifest} == {"group_000"}
    facts = project.resolve_table("orders")
    assert facts is not None and (facts.schema, facts.name) == (name, "orders")
    assert project.resolve_table("nobody") is None


def test_each_kind_of_breakage_is_pending_work_not_a_crash(cockpit_schema: str) -> None:
    name = cockpit_schema
    project = Project(CONFIG, PgSource.from_env(), schema=name)
    status = FeaturizerStatus(project)
    runs = FeaturizerRuns(project.source, name)
    Featurizer(str(CONFIG)).to_tables(name)
    with project.source.connect(autocommit=True) as conn:
        manifest = sql.Identifier(name, "customers_manifest")

        # The manifest is one column behind the config: drift.
        conn.execute(
            sql.SQL(
                "delete from {} where column_name = (select min(column_name) from {})"
            ).format(manifest, manifest)
        )
        pending = {i.detail for i in status.status().pending}
        assert any("config drift: 1 new in config" in d for d in pending), pending
        assert any(
            "1 column on disk the manifest does not describe" in d for d in pending
        ), pending

        # A group table gone: the run fails and the gauge drops.
        conn.execute(
            sql.SQL("drop table {}").format(sql.Identifier(name, "customers_group_000"))
        )
        s = status.status()
        assert any(i.level == "error" and "group_000" in i.detail for i in s.pending)
        assert runs.list()[0].state is RunState.FAILED
        assert runs.show(f"{name}.customers").stages[0].note == "table missing"
        assert {g.name: g.value for g in s.gauges}[f"{name}.customers"] == 0.0

        # A source table gone: the entity's gauge says so and pending is an error.
        conn.execute("drop table orders")
        s = status.status()
        errors = {i.name: i.detail for i in s.pending if i.level == "error"}
        assert "does not exist" in errors["orders"]
        assert {g.name: g.note for g in s.gauges}["orders rows"] == "table missing"
        assert project.resolve_table("orders") is None
