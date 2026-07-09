# coding: utf-8

"""Database execution helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import pandas as pd
import records  # type: ignore[import-untyped]
from loguru import logger

_AS_OF_DATES = "as_of_dates"


def analyze_as_of_dates(conn: Any) -> None:
    """Refresh planner statistics on the caller's ``as_of_dates`` table.

    Every generated query is ``from as_of_dates cross join lateral (…)``. A
    freshly created / never-analyzed ``as_of_dates`` has no statistics, so
    PostgreSQL assumes its ~2550-row default and can pick a catastrophic join
    plan for the lateral body — measured 40–50× slower on wide configs
    (dirtyduck / donorschoose all-agg dropped from ~300s to ~7s once analyzed).

    ``ANALYZE`` is a read-only stats refresh and **best-effort**: it is a pure
    optimization, so a caller without ANALYZE privilege must still get correct
    (if slower) results. Isolated by a SAVEPOINT so a failure cannot poison the
    caller's open transaction; failures are logged, never raised.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("savepoint featurizer_analyze")
            try:
                cur.execute(f"analyze {_AS_OF_DATES}")
                cur.execute("release savepoint featurizer_analyze")
            except Exception as exc:
                cur.execute("rollback to savepoint featurizer_analyze")
                logger.warning(
                    "ANALYZE {} skipped (planner-stats optimization only): {}",
                    _AS_OF_DATES,
                    exc,
                )
    except Exception as exc:  # e.g. autocommit connection: no transaction to savepoint
        logger.debug(
            "ANALYZE {} savepoint unavailable, skipping: {}", _AS_OF_DATES, exc
        )


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

        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(frame, on=["as_of_date", target_id], how="outer")
        return result.set_index(["as_of_date", target_id], inplace=False)
