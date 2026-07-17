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

Three contract extensions (ADR-0014), all additive:

- :class:`MultiColumnBridge` — one pass emits N value columns
  (``compute() → {pk: {col: val}}``): NER entity-type counts, a node's full
  centrality profile from one graph build.
- **Temporal snapshot sequences** — :meth:`BridgeComputer.compute_snapshots` /
  :meth:`BridgeComputer.materialize_snapshots` rebuild the model per as-of
  window on the pre-t₀ slice and key the output ``(entity, as_of_date)``, so
  non-local features (graph centrality) become an ordinary event stream the
  spine can trend. Cost is O(windows × build) — deliberate, no approximation.
- ``persist=`` on materialization (default stays session-temporary) and an
  optional ``model_vintage`` (training-cutoff date) so strict backtests can
  assert a *pretrained* model predates the cutoff — ``assert_pre_t0`` guards
  fitted models only; a pretrained snapshot trained on post-t₀ data is silent
  leakage unless its vintage is declared and checked.

This module is the orchestration-agnostic *library* (ADR-0003): it takes a live
``psycopg`` connection and does I/O when called, but schedules nothing. Wire it
into Dagster/Snakemake as a normal asset/rule upstream of the SQL run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

    #: Training-cutoff date of a *pretrained* model this bridge wraps, or
    #: ``None`` when no pretrained model is involved / the vintage is unknown.
    #: ``assert_pre_t0`` guards fitted models only — a pretrained snapshot
    #: trained on post-t₀ data is silent leakage unless this is declared and
    #: checked via :meth:`assert_model_vintage` (ADR-0014).
    model_vintage: Any = None

    def __init__(
        self,
        *,
        name: str,
        value_col: str,
        value_type: str = "numeric",
        model_vintage: Any = None,
    ):
        self.name = name
        self.value_col = value_col
        self.value_type = value_type
        if model_vintage is not None:
            self.model_vintage = model_vintage

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

    # ---- model-vintage guard (ADR-0014) -------------------------------- #

    @property
    def metadata(self) -> Dict[str, Any]:
        """Bridge identity + leakage-relevant metadata for audit trails."""
        return {
            "name": self.name,
            "value_cols": self._value_columns(),
            "model_vintage": self.model_vintage,
        }

    def assert_model_vintage(self, as_of: Any) -> None:
        """Opt-in strict check: the pretrained model must predate ``as_of``.

        Raises when the vintage is *unknown* (``None``) — a strict backtest
        cannot certify what it cannot date — or when the declared training
        cutoff is after ``as_of``.
        """
        if self.model_vintage is None:
            raise ValueError(
                f"{self.name}: model_vintage is unknown; a strict backtest "
                "requires the pretrained model's training-cutoff date. Declare "
                "model_vintage on the bridge (ADR-0014)."
            )
        if self.model_vintage > as_of:
            raise ValueError(
                f"{self.name}: pretrained model vintage "
                f"{self.model_vintage!r} is after as_of={as_of!r} — the model "
                "was trained on data not knowable at the cutoff (silent "
                "leakage, ADR-0014)."
            )

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
        persist: bool = False,
    ) -> str:
        """Compute φ and write ``output_table`` = (pk, carry_cols…, value_col).

        ``carry_cols`` are columns copied through verbatim (typically the FK to
        the parent entity and the event ``temporal_ix``) so the materialized
        table is a drop-in event stream the SQL spine can aggregate. When both
        ``causal_col`` and ``fit_before`` are given, the model is fit only on
        rows with ``causal_col <= fit_before`` and the cut is asserted.
        ``persist=True`` writes a real table (an orchestrated asset, ADR-0003)
        instead of the default session-temporary one.
        """
        select_cols: List[str] = [pk]
        for col in list(carry_cols) + list(content_cols):
            if col not in select_cols:
                select_cols.append(col)
        if causal_col and causal_col not in select_cols:
            select_cols.append(causal_col)

        rows = self._load_rows(conn, source_table, select_cols)
        fit_rows = self._fit_slice(rows, causal_col, fit_before)
        values = self.compute(rows, fit_rows=fit_rows)

        keep = [pk] + [c for c in carry_cols if c != pk]
        cols_ddl = ", ".join(f"{c} {self._carry_type(c, rows)}" for c in keep)
        value_ddl = ", ".join(
            f"{col} {typ}"
            for col, typ in zip(self._value_columns(), self._value_ddl_types(values))
        )
        n_values = len(self._value_columns())
        with conn.cursor() as cur:
            cur.execute(
                self._create_table_sql(
                    output_table, f"{cols_ddl}, {value_ddl}", persist
                )
            )
            placeholders = ", ".join(["%s"] * (len(keep) + n_values))
            payload = [
                tuple(row[c] for c in keep) + self._value_tuple(values.get(row[pk]))
                for row in rows
            ]
            cur.executemany(
                f"insert into {output_table} values ({placeholders})", payload
            )
        return output_table

    # ---- temporal snapshot sequences (ADR-0014) ------------------------ #

    def compute_snapshots(
        self,
        rows: List[Dict[str, Any]],
        *,
        as_of_dates: Sequence[Any],
        causal_col: str,
    ) -> Dict[Tuple[Any, Any], Any]:
        """Rebuild per as-of window on the pre-t₀ slice; key by (entity, as_of).

        For **non-local** φ (graph centrality: one future edge changes every
        node's score) a single ``fit_before`` cannot serve a backtest cohort
        with many as-of dates. This helper re-slices ``rows`` to
        ``causal_col <= as_of`` *per window*, asserts the boundary each time,
        and calls :meth:`compute` on the slice alone — never on a full-history
        model sliced afterwards (that leaks the future).

        Intended for bridges whose :meth:`compute` keys by *entity* (node, not
        source row); per-row content bridges gain nothing from re-computation
        per window. Cost is deliberately **O(windows × build)** — there is no
        snapshot-binning approximation. Choose cheap metrics by default and
        make expensive ones opt-in (see ``CentralityBridge``).
        """
        out: Dict[Tuple[Any, Any], Any] = {}
        for as_of in sorted(as_of_dates):
            window = [
                r
                for r in rows
                if r.get(causal_col) is not None and r[causal_col] <= as_of
            ]
            assert_pre_t0(window, as_of, causal_col)
            values = self.compute(window, fit_rows=window)
            for key, val in values.items():
                out[(key, as_of)] = val
        return out

    def materialize_snapshots(
        self,
        conn: Any,
        *,
        source_table: str,
        output_table: str,
        as_of_dates: Sequence[Any],
        causal_col: str,
        content_cols: Sequence[str] = (),
        entity_col: str = "entity_id",
        as_of_col: str = "as_of_date",
        persist: bool = False,
    ) -> str:
        """Materialize the snapshot sequence as an ordinary event stream.

        Writes ``output_table`` = (entity_col, as_of_col, value column(s)) from
        :meth:`compute_snapshots`. Declared with ``temporal_ix=as_of_col`` (see
        :meth:`emit_yaml` with ``pk=entity_col``), the table is a normal event
        stream: the spine aggregates "trend in centrality" as a window over a
        metric — no engine change. O(windows × build); see
        :meth:`compute_snapshots`.
        """
        select_cols = [c for c in content_cols]
        if causal_col not in select_cols:
            select_cols.append(causal_col)
        rows = self._load_rows(conn, source_table, select_cols)
        snapshots = self.compute_snapshots(
            rows, as_of_dates=as_of_dates, causal_col=causal_col
        )

        entity_type = self._value_sql_type([k for k, _ in snapshots.keys()])
        as_of_type = self._value_sql_type(list(as_of_dates))
        value_ddl = ", ".join(
            f"{col} {typ}"
            for col, typ in zip(self._value_columns(), self._value_ddl_types(snapshots))
        )
        n_values = len(self._value_columns())
        with conn.cursor() as cur:
            cur.execute(
                self._create_table_sql(
                    output_table,
                    f"{entity_col} {entity_type}, {as_of_col} {as_of_type}, "
                    f"{value_ddl}",
                    persist,
                )
            )
            placeholders = ", ".join(["%s"] * (2 + n_values))
            payload = [
                (entity, as_of) + self._value_tuple(val)
                for (entity, as_of), val in sorted(
                    snapshots.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][0]))
                )
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
        aggregates over ``parent_alias``. For a snapshot-sequence table pass
        ``pk=entity_col`` and ``temporal_ix=as_of_col``.
        """
        entity: Dict[str, Any] = {
            "alias": self.name,
            "table": output_table,
            "id": pk,
            "variables": self._variable_declarations(),
        }
        if temporal_ix:
            entity["temporal_ix"] = temporal_ix
        relationship = {
            "parent": {"entity": parent_alias, "key": parent_key},
            "child": {"entity": self.name, "key": fk},
        }
        return {"entity": entity, "relationship": relationship}

    # ---- helpers ------------------------------------------------------- #

    def _value_columns(self) -> List[str]:
        """The value column name(s) this bridge writes; overridden by
        :class:`MultiColumnBridge`."""
        return [self.value_col]

    def _variable_declarations(self) -> Dict[str, Dict[str, Any]]:
        """``variables:`` block for :meth:`emit_yaml`, one per value column."""
        return {self.value_col: {"type": self.value_type}}

    def _value_ddl_types(self, values: Dict[Any, Any]) -> List[str]:
        """SQL type per value column, sized from the computed values."""
        return [self._column_type(values)]

    def _value_tuple(self, value: Any) -> Tuple[Any, ...]:
        """INSERT payload fragment for one computed value (or None)."""
        return (value,)

    @staticmethod
    def _load_rows(
        conn: Any, source_table: str, select_cols: Sequence[str]
    ) -> List[Dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(f"select {', '.join(select_cols)} from {source_table}")
            names = [d.name for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

    @staticmethod
    def _fit_slice(
        rows: List[Dict[str, Any]], causal_col: Optional[str], fit_before: Any
    ) -> List[Dict[str, Any]]:
        """The pre-t₀ fit subset, boundary-asserted; all rows when uncut."""
        if not (causal_col and fit_before is not None):
            return rows
        fit_rows = [
            r
            for r in rows
            if r.get(causal_col) is not None and r[causal_col] <= fit_before
        ]
        assert_pre_t0(fit_rows, fit_before, causal_col)
        return fit_rows

    @staticmethod
    def _create_table_sql(output_table: str, cols_ddl: str, persist: bool) -> str:
        """``create table`` DDL: session-temporary by default, real when
        ``persist`` (an orchestrated asset, ADR-0003/ADR-0014)."""
        if persist:
            return f"create table {output_table} ({cols_ddl})"
        return f"create temp table {output_table} ({cols_ddl}) on commit drop"

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
    def _python_sql_type(value: Any) -> Optional[str]:
        """SQL type for one non-null Python value, or None to keep looking."""
        import datetime

        if value is None:
            return None
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

    @classmethod
    def _value_sql_type(cls, values: Sequence[Any]) -> str:
        """Best-effort SQL type from a sequence of Python values."""
        for value in values:
            typ = cls._python_sql_type(value)
            if typ is not None:
                return typ
        return "text"

    @classmethod
    def _carry_type(cls, col: str, rows: List[Dict[str, Any]]) -> str:
        """Best-effort SQL type for a carried column from its Python values."""
        return cls._value_sql_type([row.get(col) for row in rows])


class MultiColumnBridge(BridgeComputer):
    """A φ precompute whose one pass emits **many** value columns (ADR-0014).

    Subclasses implement :meth:`compute` returning ``{pk: {col: val}}`` — one
    expensive pass (a spaCy parse, a graph build) fans out into N declared
    ``value_cols``. Materialization writes one column per name;
    :meth:`emit_yaml` declares one ``Variable`` per column, so the SQL spine
    aggregates each independently. ``value_types`` maps a column to its
    variable type (default ``"numeric"``); ``"categorical"`` flows through the
    existing ADR-0007 fixed-vocabulary one-hot path unchanged.
    """

    def __init__(
        self,
        *,
        name: str,
        value_cols: Sequence[str],
        value_types: Optional[Dict[str, str]] = None,
        model_vintage: Any = None,
    ):
        if not value_cols:
            raise ValueError(f"{name}: value_cols must name at least one column")
        super().__init__(
            name=name,
            value_col=value_cols[0],
            value_type=(value_types or {}).get(value_cols[0], "numeric"),
            model_vintage=model_vintage,
        )
        self.value_cols: List[str] = list(value_cols)
        self.value_types: Dict[str, str] = {
            col: (value_types or {}).get(col, "numeric") for col in value_cols
        }

    @abstractmethod
    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        """Return ``{primary_key_value: {value_col: φ}}``.

        Missing columns in an inner dict (and missing keys entirely) become
        NULLs in the materialized table.
        """

    # ---- multi-column plumbing ----------------------------------------- #

    def _value_columns(self) -> List[str]:
        return self.value_cols

    def _variable_declarations(self) -> Dict[str, Dict[str, Any]]:
        return {col: {"type": self.value_types[col]} for col in self.value_cols}

    def _value_ddl_types(self, values: Dict[Any, Any]) -> List[str]:
        ddl: List[str] = []
        for col in self.value_cols:
            if self.value_types[col] == "numeric":
                ddl.append("double precision")
            else:
                observed = [
                    v.get(col)
                    for v in values.values()
                    if isinstance(v, dict) and v.get(col) is not None
                ]
                ddl.append(self._value_sql_type(observed))
        return ddl

    def _value_tuple(self, value: Any) -> Tuple[Any, ...]:
        if not isinstance(value, dict):
            return tuple(None for _ in self.value_cols)
        return tuple(value.get(col) for col in self.value_cols)
