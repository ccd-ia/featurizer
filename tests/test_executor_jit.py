# coding: utf-8

"""DB-free tests for ``jit = off`` around the generated queries (issue #53).

PostgreSQL compiles every expression of a wide target list before it reads a
row. Measured on the three live databases (``specs/jit-on-off/raw/``): faster in
none of nine cells, 1.05x to 7.9x slower from all-agg up, no value moved.

The contract under test:

1. ``jit_disabled`` turns ``jit`` off for a block and puts the previous value
   back — on a caller's ``connection=`` too, which ``PLANNER_TUNING`` never
   touches. ``SET LOCAL`` inside a transaction, a session ``SET`` on an
   autocommit connection (where ``SET LOCAL`` does nothing).
2. It is an optimization: it never raises, and a failure to restore never hides
   the error of the block it wrapped.
3. The block covers the TEMP-table preamble, which runs wide queries of its own.
4. ``PLANNER_TUNING`` carries ``jit = off`` for the connections featurizer opens
   (the ``records`` fast path has no psycopg connection to wrap).
"""

from typing import Any, Optional

import pytest

from featurizer.executor import (
    PLANNER_TUNING,
    QueryExecutor,
    jit_disabled,
    tuning_statements,
)


class JitCursor:
    """Records SQL, answers ``show jit`` and tracks ``set [local] jit``."""

    def __init__(self, conn: "JitConnection") -> None:
        self._conn = conn
        self._row: Optional[tuple[Any, ...]] = None
        self.description = [type("D", (), {"name": n}) for n in ("as_of_date", "id")]

    def execute(self, sql: str, *args: Any) -> None:
        conn = self._conn
        if conn.fail_on is not None and sql.startswith(conn.fail_on):
            raise RuntimeError(f"forced failure on {sql!r}")
        if conn.autocommit and sql.startswith(("savepoint", "set local")):
            raise RuntimeError(f"{sql!r} can only be used in transaction blocks")
        conn.executed.append(sql)
        if sql == "show jit":
            self._row = (conn.jit,)
        elif sql.startswith(("set local jit = ", "set jit = ")):
            conn.jit = sql.rsplit(" ", 1)[1]
        conn.jit_seen[sql] = conn.jit

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        return self._row

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [("2020-01-01", 1)]

    def __enter__(self) -> "JitCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class JitConnection:
    """psycopg-shaped connection double with a ``jit`` setting."""

    def __init__(
        self,
        jit: str = "on",
        *,
        autocommit: bool = False,
        fail_on: Optional[str] = None,
    ) -> None:
        self.jit = jit
        self.autocommit = autocommit
        self.fail_on = fail_on
        self.executed: list[str] = []
        self.jit_seen: dict[str, str] = {}  # statement -> jit in force when it ran
        self.closed = False

    def cursor(self) -> JitCursor:
        return JitCursor(self)

    def close(self) -> None:
        self.closed = True


def _jit_statements(conn: JitConnection) -> list[str]:
    return [sql for sql in conn.executed if " jit" in sql]


def test_the_block_runs_with_jit_off_and_the_value_comes_back() -> None:
    conn = JitConnection(jit="on")
    with jit_disabled(conn):
        assert conn.jit == "off"
    assert conn.jit == "on"
    assert _jit_statements(conn) == [
        "show jit",
        "set local jit = off",
        "set local jit = on",
    ]


def test_each_statement_is_savepoint_isolated_inside_a_transaction() -> None:
    conn = JitConnection(jit="on")
    with jit_disabled(conn):
        pass
    assert conn.executed == [
        "savepoint featurizer_opt",
        "show jit",
        "release savepoint featurizer_opt",
        "savepoint featurizer_opt",
        "set local jit = off",
        "release savepoint featurizer_opt",
        "savepoint featurizer_opt",
        "set local jit = on",
        "release savepoint featurizer_opt",
    ]


def test_a_connection_with_jit_already_off_is_left_alone() -> None:
    conn = JitConnection(jit="off")
    with jit_disabled(conn):
        assert conn.jit == "off"
    assert _jit_statements(conn) == ["show jit"]


def test_an_autocommit_connection_gets_a_session_set_and_no_savepoint() -> None:
    conn = JitConnection(jit="on", autocommit=True)
    with jit_disabled(conn):
        assert conn.jit == "off"
    assert conn.jit == "on"
    assert conn.executed == ["show jit", "set jit = off", "set jit = on"]


def test_a_failing_block_raises_its_own_error_and_jit_comes_back() -> None:
    conn = JitConnection(jit="on")
    with pytest.raises(ValueError, match="the block's own"):
        with jit_disabled(conn):
            raise ValueError("the block's own")
    assert conn.jit == "on"


def test_a_restore_that_fails_does_not_hide_the_blocks_error() -> None:
    # An aborted transaction refuses every statement, the restore included; the
    # rollback the caller owes then undoes SET LOCAL anyway.
    conn = JitConnection(jit="on", fail_on="set local jit = on")
    with pytest.raises(ValueError, match="the block's own"):
        with jit_disabled(conn):
            raise ValueError("the block's own")


def test_a_set_that_fails_runs_the_block_anyway_and_restores_nothing() -> None:
    conn = JitConnection(jit="on", fail_on="set local jit = off")
    ran = []
    with jit_disabled(conn):  # must not raise
        ran.append(True)
    assert ran == [True]
    assert not any(sql == "set local jit = on" for sql in conn.executed)


def test_a_setting_that_cannot_be_read_runs_the_block_untouched() -> None:
    conn = JitConnection(jit="on", fail_on="show jit")
    with jit_disabled(conn):
        assert conn.jit == "on"
    assert not any(sql.startswith("set") for sql in conn.executed)


def test_a_connection_without_cursor_support_does_not_raise() -> None:
    class NoCursor:
        def cursor(self) -> Any:
            raise RuntimeError("no cursor")

    with jit_disabled(NoCursor()):
        pass


def test_planner_tuning_carries_jit_off_for_featurizers_own_connections() -> None:
    assert ("jit", "off") in PLANNER_TUNING
    assert "set local jit = 'off'" in tuning_statements()


def test_materialized_path_turns_jit_off_on_a_callers_connection() -> None:
    conn = JitConnection(jit="on")
    preamble = "create temp table shard_0 on commit drop as select 1"
    QueryExecutor().to_dataframe_materialized(
        preamble_ddl=[preamble],
        group_queries={"group_000": "select 1", "group_001": "select 2"},
        target_id="id",
        connection=conn,
    )
    # The preamble runs wide queries of its own, so it is inside the block too.
    assert conn.jit_seen[preamble] == "off"
    assert conn.jit_seen["select 1"] == "off"
    assert conn.jit_seen["select 2"] == "off"
    assert conn.jit == "on"  # the caller gets their value back
    assert conn.executed.index("set local jit = on") > conn.executed.index("select 2")
    # jit is the only setting a caller's connection ever sees changed.
    assert not any(
        f"set local {name}" in sql
        for sql in conn.executed
        for name, _ in PLANNER_TUNING
        if name != "jit"
    )
    assert not conn.closed


def test_materialized_path_restores_jit_when_a_group_query_fails() -> None:
    conn = JitConnection(jit="on", fail_on="select 2")
    with pytest.raises(RuntimeError, match="materialized query execution failed"):
        QueryExecutor().to_dataframe_materialized(
            preamble_ddl=[],
            group_queries={"group_000": "select 1", "group_001": "select 2"},
            target_id="id",
            connection=conn,
        )
    assert conn.jit == "on"
