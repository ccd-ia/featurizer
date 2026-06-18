# coding: utf-8

"""Database execution helpers."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import pandas as pd
import records  # type: ignore[import-untyped]
from loguru import logger


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
