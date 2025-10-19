# coding: utf-8

"""Database execution helpers."""

import records


class QueryExecutor:
    """Handles optional execution of rendered SQL queries."""

    def __init__(self, database_factory=None) -> None:
        self._database_factory = database_factory or records.Database

    def to_dataframe(self, query: str, target_id: str):
        db = self._database_factory()
        rows = db.query(query)
        df = rows.export('df')
        return df.set_index(['as_of_date', target_id], inplace=False)
