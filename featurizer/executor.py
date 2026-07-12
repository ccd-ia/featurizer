# coding: utf-8

"""Database execution helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd
import records  # type: ignore[import-untyped]
from loguru import logger

_AS_OF_DATES = "as_of_dates"

# Conservative planner/memory tuning for the generated queries' wide multi-way
# CTE joins (an all-aggregator config merge-joins ~38 CTEs). Measured on the
# dirtyduck all-agg config: ~1.4× (359.3s → 259.9s). Values are deliberately
# modest: the aggressive variant (work_mem 256MB, collapse limits 30, geqo off)
# crashed the backend — exhaustive planning of a 38-way join explodes — so the
# collapse limits stay below geqo_threshold's reach and geqo stays ON. All three
# GUCs are USERSET (any role may SET them), and ``SET LOCAL`` is scoped to the
# current transaction, so nothing leaks past featurizer's own work.
PLANNER_TUNING: tuple[tuple[str, str], ...] = (
    ("work_mem", "64MB"),
    ("join_collapse_limit", "20"),
    ("from_collapse_limit", "20"),
)


def tuning_statements() -> list[str]:
    """The ``SET LOCAL`` statements implementing :data:`PLANNER_TUNING`."""
    return [f"set local {name} = '{value}'" for name, value in PLANNER_TUNING]


def _run_isolated(conn: Any, statements: Sequence[str], purpose: str) -> None:
    """Run pure-optimization statements under a SAVEPOINT; never raise.

    A failure rolls back to the savepoint so it cannot poison the caller's open
    transaction, and is logged instead of raised — these statements are
    optimizations, so a caller without the needed privilege must still get
    correct (if slower) results.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("savepoint featurizer_opt")
            try:
                for statement in statements:
                    cur.execute(statement)
                cur.execute("release savepoint featurizer_opt")
            except Exception as exc:
                cur.execute("rollback to savepoint featurizer_opt")
                logger.warning("{} skipped (optimization only): {}", purpose, exc)
    except Exception as exc:  # e.g. autocommit connection: no transaction to savepoint
        logger.debug("{} savepoint unavailable, skipping: {}", purpose, exc)


def analyze_as_of_dates(conn: Any) -> None:
    """Refresh planner statistics on the caller's ``as_of_dates`` table.

    Every generated query is ``from as_of_dates cross join lateral (…)``. A
    freshly created / never-analyzed ``as_of_dates`` has no statistics, so
    PostgreSQL assumes its ~2550-row default and can pick a catastrophic join
    plan for the lateral body — measured 40–50× slower on wide configs
    (dirtyduck / donorschoose all-agg dropped from ~300s to ~7s once analyzed).

    ``ANALYZE`` is a read-only stats refresh and **best-effort**: see
    :func:`_run_isolated` (savepoint-isolated, logged, never raised).
    """
    _run_isolated(
        conn,
        [f"analyze {_AS_OF_DATES}"],
        f"ANALYZE {_AS_OF_DATES} (planner-stats optimization)",
    )


def apply_planner_tuning(conn: Any) -> None:
    """``SET LOCAL`` the :data:`PLANNER_TUNING` values on a psycopg connection.

    Only ever called on connections featurizer opened itself — a caller's
    ``connection=`` is never tuned, because ``SET LOCAL`` would stay in force
    for the remainder of *their* open transaction. Best-effort and
    savepoint-isolated like :func:`analyze_as_of_dates` (rolling back to a
    savepoint also cancels any partially-applied ``SET LOCAL``).
    """
    _run_isolated(conn, tuning_statements(), "planner/memory tuning")


class QueryExecutor:
    """Handles optional execution of rendered SQL queries."""

    def __init__(self, database_factory: Optional[Callable[[], Any]] = None) -> None:
        """Initialize executor with optional database factory.

        Args:
            database_factory: Callable that returns a database connection.
                            Defaults to records.Database if not provided.
        """
        self._database_factory = database_factory or records.Database

    def to_dataframe(self, query: str, target_id: str) -> pd.DataFrame:
        """Execute query and return results as a pandas DataFrame.

        Args:
            query: SQL query string to execute
            target_id: Name of the target entity's ID column for indexing

        Returns:
            DataFrame indexed by ['as_of_date', target_id]

        Raises:
            RuntimeError: If the database rejects the rendered query. The full
                SQL is logged at error level so the failing CTE name can be
                traced back to the planner builder that emitted it.
        """
        db = self._database_factory()
        try:
            # Planner-stats optimization (best-effort): give the caller's
            # as_of_dates real statistics before the lateral-join query. See
            # analyze_as_of_dates. records auto-commits per query, so a failure
            # here is isolated and non-fatal.
            try:
                db.query(f"analyze {_AS_OF_DATES}")
            except Exception as exc:
                logger.warning(
                    "ANALYZE {} skipped (planner-stats optimization only): {}",
                    _AS_OF_DATES,
                    exc,
                )
            # Planner/memory tuning: SET LOCAL is transaction-scoped, and
            # records' Database.query opens a fresh pooled connection per call,
            # so the tuning must share ONE connection (and its SQLAlchemy 2
            # autobegun transaction) with the main query. records' own
            # Database.transaction() context manager swallows exceptions (bare
            # except without re-raise), which would break this method's
            # error-reporting contract — so hold a records Connection directly.
            # The GUCs are USERSET, so the SETs cannot fail on privilege.
            if hasattr(db, "get_connection"):
                conn = db.get_connection()
                try:
                    for statement in tuning_statements():
                        conn.query(statement)
                    rows = conn.query(query)
                    df = rows.export("df")
                finally:
                    # SELECT-only transaction: close (implicit rollback) is fine
                    # and returns the connection to the pool untuned.
                    conn.close()
            else:  # duck-typed database_factory without records' Connection API
                rows = db.query(query)
                df = rows.export("df")
        except Exception as exc:
            logger.error(
                "Featurizer query execution failed: {}\n--- rendered SQL ---\n{}",
                exc,
                query,
            )
            raise RuntimeError(
                f"Featurizer query execution failed ({exc}). The rendered SQL is "
                "logged above; look up the CTE named in the database error and "
                "trace it back to the planner builder that emitted it."
            ) from exc
        return df.set_index(["as_of_date", target_id], inplace=False)

    def to_dataframe_materialized(
        self,
        *,
        preamble_ddl: list[str],
        group_queries: Mapping[str, str],
        target_id: str,
        connection: Any = None,
        key_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Execute a grouped / temp-table-materialized query on ONE connection.

        The ``records`` fast path opens a fresh connection per query, so the
        session ``TEMP`` tables created by the materialization preamble (issue #7)
        would not survive across the group queries. This path runs the preamble +
        every group query on a single psycopg connection (non-autocommit, so
        ``ON COMMIT DROP`` shards live for the transaction), then re-joins the
        column groups on ``(as_of_date, target_id)`` into one frame — the same
        indexed contract as :meth:`to_dataframe`.

        Args:
            preamble_ddl: ``CREATE TEMP TABLE`` statements to run first (may be []).
            group_queries: ``group_id -> SQL`` (each leads with as_of_date + id).
            target_id: The target entity's id column (the re-join key).
            connection: An open psycopg connection to reuse (e.g. the integration
                harness). When ``None``, one is built from the environment and
                closed afterwards.
            key_columns: The full identifier tuple every group leads with
                (``GroupedQueries.key_columns``). A target that carries extra
                identifier columns beyond its id (e.g. relationship keys such
                as donorschoose's ``schoolid``/``teacher_acctid``) repeats them
                in *every* group, so the re-join must merge on all of them —
                merging on ``(as_of_date, id)`` alone duplicates the carried
                columns and pandas raises ``MergeError`` at the third group.
                Defaults to ``["as_of_date", target_id]``.

        Returns:
            DataFrame indexed by ``['as_of_date', target_id]``.
        """
        from .arrow import default_connection

        own_connection = connection is None
        conn = connection if connection is not None else default_connection()
        try:
            if preamble_ddl:
                with conn.cursor() as cur:
                    for ddl in preamble_ddl:
                        cur.execute(ddl)
            # Planner-stats optimization: analyze the caller's as_of_dates once so
            # the lateral-join plan is not built for the ~2550-row no-stats default
            # (40–50× on wide configs). Best-effort + savepoint-isolated.
            analyze_as_of_dates(conn)
            # Planner/memory tuning — but never on a caller's connection=, where
            # SET LOCAL would outlive us inside their open transaction.
            if own_connection:
                apply_planner_tuning(conn)
            frames: list[pd.DataFrame] = []
            for sql in group_queries.values():
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cols: list[str] = [desc.name for desc in cur.description]
                    rows = cur.fetchall()
                frames.append(pd.DataFrame(rows, columns=pd.Index(cols)))
        except Exception as exc:
            logger.error("Featurizer materialized execution failed: {}", exc)
            raise RuntimeError(
                f"Featurizer materialized query execution failed ({exc}). The "
                "preamble + group queries are logged above; trace the CTE named in "
                "the database error back to the planner builder that emitted it."
            ) from exc
        finally:
            if own_connection:
                conn.close()

        join_keys = list(key_columns) if key_columns else ["as_of_date", target_id]
        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(frame, on=join_keys, how="outer")
        return result.set_index(["as_of_date", target_id], inplace=False)
