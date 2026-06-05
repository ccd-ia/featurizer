# coding: utf-8

"""Database execution helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd
import records  # type: ignore[import-untyped]


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
        """
        db = self._database_factory()
        rows = db.query(query)
        df = rows.export("df")
        return df.set_index(["as_of_date", target_id], inplace=False)
