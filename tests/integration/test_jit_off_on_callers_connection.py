"""``jit`` is off while featurizer runs on a caller's connection, and back after.

Issue #53. A consumer runs featurizer on its own connection, because that is
where its TEMP ``as_of_dates`` lives, and ``PLANNER_TUNING`` never touches that
connection. So the measurement's gain (``specs/jit-on-off/raw/``: up to 7.9x on a
wide config, faster with JIT in no cell) reaches a consumer only if ``jit`` is
turned off there too, and the caller's own value has to survive the call.

The integration tier runs with ``PGOPTIONS="-c jit=off"`` (``just
test-integration``), which would make every assertion here true by accident. Each
test therefore turns ``jit`` ON in the caller's transaction first.

What is observed is the setting in force when each statement ran, read with
``show jit`` on the same connection right before the statement is sent.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest
import yaml

from featurizer import Featurizer

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

_SCHEMA = "fz_jit_test"
#: Statements that are the mechanism itself, or transaction control around it.
_NOT_WORK = ("show ", "set ", "savepoint ", "release ", "rollback ")


class _SpyCursor:
    def __init__(self, cursor: Any, spy: "_SpyConnection") -> None:
        self._cursor = cursor
        self._spy = spy

    def _observe(self, sql: Any) -> None:
        text = sql if isinstance(sql, str) else str(sql)
        if text.lstrip().lower().startswith(_NOT_WORK):
            return
        with self._spy.real.cursor() as probe:
            probe.execute("show jit")
            self._spy.seen.append((probe.fetchone()[0], " ".join(text.split())[:70]))

    def execute(self, sql: Any, *args: Any, **kwargs: Any) -> Any:
        self._observe(sql)
        return self._cursor.execute(sql, *args, **kwargs)

    def copy(self, sql: Any, *args: Any, **kwargs: Any) -> Any:
        self._observe(sql)
        return self._cursor.copy(sql, *args, **kwargs)

    def __enter__(self) -> "_SpyCursor":
        self._cursor.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._cursor.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _SpyConnection:
    """The caller's connection, recording the ``jit`` each statement ran under."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.seen: list[tuple[str, str]] = []

    def cursor(self, *args: Any, **kwargs: Any) -> _SpyCursor:
        return _SpyCursor(self.real.cursor(*args, **kwargs), self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


def _seed(conn: Any) -> None:
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,), (2,)])
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("customer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [
            (1, 1, "2023-06-01", 10.0),
            (2, 1, "2023-07-01", 20.0),
            (3, 2, "2023-08-01", 30.0),
        ],
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])


def _featurizer(*, materialized: bool) -> Featurizer:
    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["count", "sum", "mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    if materialized:
        # threshold=1 forces the child chain into TEMP shards: a preamble to cover.
        return Featurizer(path, validate=False, materialize_threshold=1)
    return Featurizer(path, validate=False)


def _to_dataframe(f: Featurizer, conn: Any, tmp_path: Any) -> None:
    f.to_dataframe(connection=conn)


def _to_arrow(f: Featurizer, conn: Any, tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    f.to_arrow(connection=conn)


def _to_parquet(f: Featurizer, conn: Any, tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    f.to_parquet(str(tmp_path / "features.parquet"), connection=conn)


def _to_tables(f: Featurizer, conn: Any, tmp_path: Any) -> None:
    f.to_tables(_SCHEMA, connection=conn)


def _jit(conn: Any) -> str:
    with conn.cursor() as cur:
        cur.execute("show jit")
        return cur.fetchone()[0]


@pytest.mark.parametrize("materialized", [False, True], ids=["inline", "temp-shards"])
@pytest.mark.parametrize(
    "run",
    [_to_dataframe, _to_arrow, _to_parquet, _to_tables],
    ids=["to_dataframe", "to_arrow", "to_parquet", "to_tables"],
)
def test_every_statement_runs_with_jit_off_and_the_callers_value_comes_back(
    pg_conn, tmp_path, run, materialized
) -> None:
    _seed(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute("set local jit = on")
    f = _featurizer(materialized=materialized)
    if materialized:
        assert f._grouped().materialization is not None, "expected a preamble"

    spy = _SpyConnection(pg_conn)
    run(f, spy, tmp_path)

    assert spy.seen, "the spy saw no statement; the path no longer uses cursor()"
    under_jit = [text for jit, text in spy.seen if jit != "off"]
    assert under_jit == [], f"ran with jit on: {under_jit}"
    if materialized:
        assert any("create temp table" in text.lower() for _, text in spy.seen)
    # Same transaction, after the call: the caller's value, not featurizer's.
    assert _jit(pg_conn) == "on"


def test_a_caller_who_already_has_jit_off_keeps_it_off(pg_conn, tmp_path) -> None:
    _seed(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute("set local jit = off")
    _to_dataframe(_featurizer(materialized=False), pg_conn, tmp_path)
    assert _jit(pg_conn) == "off"


def test_a_failing_query_leaves_nothing_behind_after_the_rollback(
    pg_conn, tmp_path
) -> None:
    # No ``orders`` table: the group query fails and aborts the transaction, so
    # the restore is refused with everything else. The rollback the caller owes
    # is what undoes SET LOCAL.
    create_temp_table(pg_conn, "customers", [("customer_id", "int")], [(1,)])
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )
    before = _jit(pg_conn)
    with pytest.raises(RuntimeError, match="orders"):
        _to_dataframe(_featurizer(materialized=False), pg_conn, tmp_path)
    pg_conn.rollback()
    assert _jit(pg_conn) == before
