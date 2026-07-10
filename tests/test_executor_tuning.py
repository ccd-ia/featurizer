# coding: utf-8

"""DB-free tests for the executor's planner/memory tuning (SET LOCAL defaults).

The contract under test:

1. ``apply_planner_tuning`` issues the ``PLANNER_TUNING`` values as
   ``SET LOCAL`` under a savepoint, and never raises (pure optimization).
2. The materialized path tunes ONLY featurizer-owned connections — a caller's
   ``connection=`` is never mutated (``SET LOCAL`` would stay in force for the
   remainder of the caller's open transaction).
3. The records fast path runs the SETs and the main query on ONE held
   connection (SET LOCAL is transaction-scoped; a per-query pooled connection
   would silently drop the tuning).
"""

from typing import Any, Optional

import pandas as pd

from featurizer.executor import (
    PLANNER_TUNING,
    QueryExecutor,
    apply_planner_tuning,
    tuning_statements,
)


class FakeCursor:
    """Records executed SQL; optionally fails on a chosen statement prefix."""

    def __init__(self, executed: list[str], fail_on: Optional[str] = None) -> None:
        self._executed = executed
        self._fail_on = fail_on
        self.description = [type("D", (), {"name": n}) for n in ("as_of_date", "id")]

    def execute(self, sql: str, *args: Any) -> None:
        if self._fail_on is not None and sql.startswith(self._fail_on):
            raise RuntimeError(f"forced failure on {sql!r}")
        self._executed.append(sql)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [("2020-01-01", 1)]

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeConnection:
    """psycopg-shaped connection double."""

    def __init__(self, fail_on: Optional[str] = None) -> None:
        self.executed: list[str] = []
        self._fail_on = fail_on
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.executed, self._fail_on)

    def close(self) -> None:
        self.closed = True


def test_tuning_statements_render_the_documented_values() -> None:
    statements = tuning_statements()
    assert statements == [
        f"set local {name} = '{value}'" for name, value in PLANNER_TUNING
    ]
    assert "set local work_mem = '64MB'" in statements
    assert "set local join_collapse_limit = '20'" in statements
    assert "set local from_collapse_limit = '20'" in statements


def test_apply_planner_tuning_is_savepoint_isolated() -> None:
    conn = FakeConnection()
    apply_planner_tuning(conn)
    assert conn.executed[0] == "savepoint featurizer_opt"
    assert conn.executed[1:-1] == tuning_statements()
    assert conn.executed[-1] == "release savepoint featurizer_opt"


def test_apply_planner_tuning_failure_rolls_back_and_does_not_raise() -> None:
    conn = FakeConnection(fail_on="set local")
    apply_planner_tuning(conn)  # must not raise
    assert conn.executed == [
        "savepoint featurizer_opt",
        "rollback to savepoint featurizer_opt",
    ]


def test_apply_planner_tuning_without_cursor_support_does_not_raise() -> None:
    class NoCursor:
        def cursor(self) -> Any:
            raise RuntimeError("autocommit connection: no transaction")

    apply_planner_tuning(NoCursor())  # must not raise


def test_materialized_path_never_tunes_a_caller_connection() -> None:
    conn = FakeConnection()
    df = QueryExecutor().to_dataframe_materialized(
        preamble_ddl=[],
        group_queries={"group_000": "select 1"},
        target_id="id",
        connection=conn,
    )
    assert not any("set local" in sql for sql in conn.executed)
    # The stats refresh is connection-agnostic (ANALYZE is global) and still runs.
    assert any(sql.startswith("analyze") for sql in conn.executed)
    assert not conn.closed  # caller's connection stays open
    assert list(df.index.names) == ["as_of_date", "id"]


def test_materialized_path_tunes_its_own_connection(monkeypatch: Any) -> None:
    conn = FakeConnection()
    import featurizer.arrow as arrow_mod

    monkeypatch.setattr(arrow_mod, "default_connection", lambda: conn)
    QueryExecutor().to_dataframe_materialized(
        preamble_ddl=[],
        group_queries={"group_000": "select 1"},
        target_id="id",
        connection=None,
    )
    for statement in tuning_statements():
        assert statement in conn.executed
    # Tuning is applied before any group query runs.
    assert conn.executed.index(tuning_statements()[0]) < conn.executed.index("select 1")
    assert conn.closed


class FakeRecordCollection:
    def export(self, kind: str) -> pd.DataFrame:
        assert kind == "df"
        return pd.DataFrame({"as_of_date": ["2020-01-01"], "id": [1]})


class FakeRecordsConnection:
    """records.Connection double: one held connection, ordered query log."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.closed = False

    def query(self, sql: str, *args: Any, **kwargs: Any) -> FakeRecordCollection:
        self._log.append(f"conn:{sql}")
        return FakeRecordCollection()

    def close(self) -> None:
        self.closed = True


class FakeRecordsDatabase:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.connection = FakeRecordsConnection(self.log)

    def query(self, sql: str, *args: Any, **kwargs: Any) -> FakeRecordCollection:
        self._log_db(sql)
        return FakeRecordCollection()

    def _log_db(self, sql: str) -> None:
        self.log.append(f"db:{sql}")

    def get_connection(self) -> FakeRecordsConnection:
        return self.connection


def test_records_path_tunes_and_queries_on_one_held_connection() -> None:
    db = FakeRecordsDatabase()
    df = QueryExecutor(database_factory=lambda: db).to_dataframe(
        "select * from features", target_id="id"
    )
    conn_calls = [c for c in db.log if c.startswith("conn:")]
    assert conn_calls[:-1] == [f"conn:{s}" for s in tuning_statements()]
    assert conn_calls[-1] == "conn:select * from features"
    assert db.connection.closed
    # ANALYZE stays on its own auto-commit call (best-effort, must not poison
    # the query transaction if the role lacks the privilege).
    assert "db:analyze as_of_dates" in db.log
    assert list(df.index.names) == ["as_of_date", "id"]


def test_records_path_falls_back_without_get_connection() -> None:
    class MinimalDb:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def query(self, sql: str, *args: Any, **kwargs: Any) -> FakeRecordCollection:
            self.queries.append(sql)
            return FakeRecordCollection()

    db = MinimalDb()
    QueryExecutor(database_factory=lambda: db).to_dataframe(
        "select * from features", target_id="id"
    )
    assert db.queries == ["analyze as_of_dates", "select * from features"]
