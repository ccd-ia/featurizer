# coding: utf-8

"""The φ-bridge: a precompute companion for non-SQL feature families.

Featurizer's SQL spine synthesizes anything expressible as point-in-time-correct
SQL aggregation. Some feature families are *not* — NER counts, sentence
embeddings, graph centrality, fitted sequence models. The φ-bridge is a thin
precompute layer for exactly those: heavy Python computes a value φ per source
row (a model, when one is needed, fit on *pre-t₀* rows only), materializes the
result back into PostgreSQL as an ordinary column, and emits a config fragment
declaring that column as a ``Variable``. The existing SQL spine then aggregates
it with its normal ``<= aod.as_of_date`` causal bound — no second feature engine.

The single hard invariant is the **causal boundary** (ADR-0001): any model is fit
only on rows knowable as-of the cutoff. ``assert_pre_t0`` enforces it fail-fast;
the per-row transform itself reads only the row's own content, so temporal
correctness is then the spine's standard aggregation bound.

This module is the orchestration-agnostic *library* (ADR-0003): it takes a live
``psycopg`` connection and does I/O when called, but schedules nothing. Wire it
into Dagster/Snakemake as a normal asset/rule upstream of the SQL run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


def assert_pre_t0(rows: Sequence[Dict[str, Any]], as_of: Any, causal_col: str) -> None:
    """Fail-fast if any row is *not* knowable as-of ``as_of``.

    A row is knowable when ``row[causal_col] <= as_of`` (the same inclusive cut
    the SQL spine uses). A model fit on a row with ``row[causal_col] > as_of``
    would leak the future into the learned representation — the exact failure
    this guard exists to make loud rather than silent.
    """
    offenders = [
        row[causal_col]
        for row in rows
        if row.get(causal_col) is not None and row[causal_col] > as_of
    ]
    if offenders:
        raise ValueError(
            f"φ-bridge causal boundary violated: {len(offenders)} fit row(s) have "
            f"{causal_col} > as_of={as_of!r} (e.g. {offenders[0]!r}). Fit only on "
            f"rows knowable as-of the cutoff."
        )


class BridgeComputer(ABC):
    """Base class for a φ precompute that materializes one column.

    Subclasses implement :meth:`compute`, mapping the source rows to a
    ``{primary_key: φ}`` dict (optionally fitting a model on the pre-t₀
    ``fit_rows`` first). The base class drives loading, the causal guard, and
    writing the output table.
    """

    #: ``"numeric"`` (a scalar column) or ``"vector"`` (a pgvector column).
    value_type: str = "numeric"

    def __init__(self, *, name: str, value_col: str, value_type: str = "numeric"):
        self.name = name
        self.value_col = value_col
        self.value_type = value_type

    # ---- subclass hook ------------------------------------------------- #

    @abstractmethod
    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Any]:
        """Return ``{primary_key_value: φ}`` for the rows to score.

        ``rows`` are every source row; ``fit_rows`` are the pre-t₀ subset any
        model must be fit on (already causal-guarded by the caller). A returned
        key absent from ``rows`` is ignored; a row absent from the result gets a
        NULL φ.
        """

    # ---- orchestration ------------------------------------------------- #

    def materialize(
        self,
        conn: Any,
        *,
        source_table: str,
        pk: str,
        output_table: str,
        carry_cols: Sequence[str] = (),
        content_cols: Sequence[str] = (),
        causal_col: Optional[str] = None,
        fit_before: Any = None,
    ) -> str:
        """Compute φ and write ``output_table`` = (pk, carry_cols…, value_col).

        ``carry_cols`` are columns copied through verbatim (typically the FK to
        the parent entity and the event ``temporal_ix``) so the materialized
        table is a drop-in event stream the SQL spine can aggregate. When both
        ``causal_col`` and ``fit_before`` are given, the model is fit only on
        rows with ``causal_col <= fit_before`` and the cut is asserted.
        """
        select_cols: List[str] = [pk]
        for col in list(carry_cols) + list(content_cols):
            if col not in select_cols:
                select_cols.append(col)
        if causal_col and causal_col not in select_cols:
            select_cols.append(causal_col)

        with conn.cursor() as cur:
            cur.execute(f"select {', '.join(select_cols)} from {source_table}")
            names = [d.name for d in cur.description]
            rows = [dict(zip(names, r)) for r in cur.fetchall()]

        if causal_col and fit_before is not None:
            fit_rows = [
                r
                for r in rows
                if r.get(causal_col) is not None and r[causal_col] <= fit_before
            ]
            assert_pre_t0(fit_rows, fit_before, causal_col)
        else:
            fit_rows = rows

        values = self.compute(rows, fit_rows=fit_rows)

        keep = [pk] + [c for c in carry_cols if c != pk]
        col_type = self._column_type(values)
        cols_ddl = ", ".join(f"{c} {self._carry_type(c, rows)}" for c in keep)
        with conn.cursor() as cur:
            cur.execute(
                f"create temp table {output_table} "
                f"({cols_ddl}, {self.value_col} {col_type}) on commit drop"
            )
            placeholders = ", ".join(["%s"] * (len(keep) + 1))
            payload = [
                tuple(row[c] for c in keep) + (values.get(row[pk]),) for row in rows
            ]
            cur.executemany(
                f"insert into {output_table} values ({placeholders})", payload
            )
        return output_table

    def emit_yaml(
        self,
        *,
        output_table: str,
        pk: str,
        parent_alias: str,
        parent_key: str,
        fk: str,
        temporal_ix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Config fragment declaring the materialized table as an entity.

        Returns ``{"entity": {...}, "relationship": {...}}`` ready to splice into
        a featurizer config: the φ column is a ``Variable`` the SQL spine then
        aggregates over ``parent_alias``.
        """
        entity: Dict[str, Any] = {
            "alias": self.name,
            "table": output_table,
            "id": pk,
            "variables": {self.value_col: {"type": self.value_type}},
        }
        if temporal_ix:
            entity["temporal_ix"] = temporal_ix
        relationship = {
            "parent": {"entity": parent_alias, "key": parent_key},
            "child": {"entity": self.name, "key": fk},
        }
        return {"entity": entity, "relationship": relationship}

    # ---- helpers ------------------------------------------------------- #

    def _column_type(self, values: Dict[Any, Any]) -> str:
        if self.value_type == "vector":
            dim = next((len(v) for v in values.values() if v is not None), None)
            if dim is None:
                raise ValueError(
                    f"{self.name}: vector φ produced no non-null values, cannot "
                    "size the pgvector column"
                )
            return f"vector({dim})"
        return "double precision"

    @staticmethod
    def _carry_type(col: str, rows: List[Dict[str, Any]]) -> str:
        """Best-effort SQL type for a carried column from its Python values."""
        import datetime

        for row in rows:
            value = row.get(col)
            if value is None:
                continue
            if isinstance(value, bool):
                return "boolean"
            if isinstance(value, int):
                return "bigint"
            if isinstance(value, float):
                return "double precision"
            if isinstance(value, datetime.datetime):
                return "timestamp"
            if isinstance(value, datetime.date):
                return "date"
            return "text"
        return "text"
