# coding: utf-8

"""Database execution helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional

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
