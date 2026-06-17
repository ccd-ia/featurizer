# coding: utf-8

"""Arrow / Parquet output for featurizer, with no pandas round-trip.

The :class:`~featurizer.executor.QueryExecutor` path materializes the whole
result set as a pandas DataFrame (``rows.export("df")``) and is the right tool
for notebook / EDA work. For handing a feature matrix to a training pipeline we
want two extra guarantees that pandas does not give cheaply:

1. **NULL fidelity.** A NULL in the ``(as_of_date × entity)`` matrix means "no
   qualifying events in the window" and is predictive signal. pandas coerces
   integer/boolean columns with NULLs to ``float`` + ``NaN``, conflating "no
   data" with "not-a-number". Arrow keeps a real null bitmap per column.
2. **No full-frame materialization in pandas.** We stream the result out of
   PostgreSQL with binary ``COPY`` and decode it column-by-column into Arrow.

This module is optional: it needs the ``[parquet]`` extra (``pyarrow``).
``psycopg`` is a core dependency, but the imports are kept lazy so importing
``featurizer`` never pulls Arrow in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa


def _require_pyarrow() -> "Any":
    """Import pyarrow, or raise a clear, install-ready ImportError.

    Mirrors the optional-dependency guard used by the ``[viz]`` and ``[bridge]``
    modules so the core package works without the ``[parquet]`` extra.
    """
    try:
        import pyarrow as pa  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "Arrow/Parquet output requires pyarrow. Install it with: "
            "pip install 'featurizer[parquet]' (or: uv sync --extra parquet)."
        ) from exc
    return pa


def _require_psycopg() -> "Any":
    """Import psycopg (a core dependency), failing fast with context if absent."""
    try:
        import psycopg  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - psycopg is declared in pyproject
        raise ImportError(
            "Arrow/Parquet output needs psycopg (a core featurizer dependency). "
            "Re-run `uv sync` to install it."
        ) from exc
    return psycopg


# PostgreSQL type OID -> pyarrow type factory. Only the OIDs featurizer's
# generated SQL can actually emit are listed; anything else falls back to a
# string column (lossless for inspection, and the caller sees the gap rather
# than a silent type coercion). decimal128 precision/scale for ``numeric`` is a
# generous default — Postgres unconstrained ``numeric`` has no fixed scale.
def _oid_arrow_type(oid: int, pa: "Any") -> Optional["pa.DataType"]:
    mapping = {
        16: pa.bool_(),  # bool
        20: pa.int64(),  # int8 / bigint
        21: pa.int16(),  # int2 / smallint
        23: pa.int32(),  # int4 / integer
        25: pa.string(),  # text
        700: pa.float32(),  # float4 / real
        701: pa.float64(),  # float8 / double precision
        1042: pa.string(),  # bpchar
        1043: pa.string(),  # varchar
        1082: pa.date32(),  # date
        1114: pa.timestamp("us"),  # timestamp without time zone
        1184: pa.timestamp("us", tz="UTC"),  # timestamptz
        1700: pa.decimal128(38, 9),  # numeric (computed aggregates land here)
        2950: pa.string(),  # uuid
    }
    return mapping.get(oid)


class ArrowExporter:
    """Runs a rendered query and decodes the result into Arrow, no pandas hop.

    The full result set is streamed out of PostgreSQL via binary ``COPY`` and
    decoded with psycopg's own per-OID loaders (so a SQL NULL becomes a Python
    ``None``, never ``NaN``), then assembled column-by-column into a
    :class:`pyarrow.Table` with a schema derived from the cursor description.
    """

    def __init__(self, connection_factory: Optional[Callable[[], Any]] = None) -> None:
        """Initialize the exporter.

        Args:
            connection_factory: Callable returning an open psycopg connection.
                Defaults to a connection built from ``DATABASE_URL`` / ``PG*``
                via :func:`featurizer.arrow.default_connection`. The integration
                harness passes an existing connection (so session ``TEMP`` tables
                referenced by the query resolve) by supplying its own factory or
                using :meth:`to_arrow`'s ``connection`` argument.
        """
        self._connection_factory = connection_factory

    def to_arrow(
        self,
        query: str,
        *,
        connection: Optional[Any] = None,
        numeric_as_float: bool = True,
    ) -> "pa.Table":
        """Execute ``query`` and return a :class:`pyarrow.Table`.

        Args:
            query: SQL query string to execute (the rendered featurizer query).
            connection: An already-open psycopg connection to run ``COPY`` on.
                Required when the query references session ``TEMP`` tables. When
                ``None``, a connection is built from the factory / environment
                and closed afterwards.
            numeric_as_float: Cast PostgreSQL ``numeric`` columns (computed
                aggregates such as ``AVG``/``STDDEV``) to ``float64`` for an
                ML-ready, ``to_dataframe``-comparable matrix. Set ``False`` to
                keep exact ``decimal128`` values. NULLs are preserved either way.

        Returns:
            A pyarrow.Table. ``as_of_date`` and the target id are ordinary
            leading columns (no index), and SQL NULLs are Arrow nulls.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
            RuntimeError: If the database rejects the query. The full SQL is
                logged at error level so the failing CTE can be traced back to
                the planner builder that emitted it (same contract as
                :meth:`QueryExecutor.to_dataframe`).
        """
        pa = _require_pyarrow()

        own_connection = connection is None
        conn = connection if connection is not None else self._open_connection()
        try:
            table = self._copy_to_arrow(conn, query, pa, numeric_as_float)
        except Exception as exc:
            logger.error(
                "Featurizer Arrow export failed: {}\n--- rendered SQL ---\n{}",
                exc,
                query,
            )
            raise RuntimeError(
                f"Featurizer Arrow export failed ({exc}). The rendered SQL is "
                "logged above; look up the CTE named in the database error and "
                "trace it back to the planner builder that emitted it."
            ) from exc
        finally:
            if own_connection:
                conn.close()
        return table

    def open_connection(self) -> Any:
        """Open a connection from the factory / environment.

        Public so callers that run several queries (e.g. the column-group
        sharding path) can open one connection, reuse it across every group, and
        close it once. Mirrors what :meth:`to_arrow` does internally when no
        connection is supplied.
        """
        return self._open_connection()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _open_connection(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory()
        return default_connection()

    @staticmethod
    def _copy_to_arrow(
        conn: Any, query: str, pa: "Any", numeric_as_float: bool
    ) -> "pa.Table":
        """COPY the query out in binary and build an Arrow table column-wise."""
        with conn.cursor() as cur:
            # One describe pass to learn column names + type OIDs. We rely on
            # COPY for the actual data transfer, but COPY's binary stream carries
            # no column metadata, so the description from a prepared statement is
            # how we know the per-column types to decode and the schema to build.
            cur.execute(query)
            columns = [desc.name for desc in cur.description]
            oids = [desc.type_code for desc in cur.description]

            decoded_rows: list[tuple[Any, ...]] = []
            copy_sql = f"copy ({query}) to stdout (format binary)"
            with cur.copy(copy_sql) as copy:
                copy.set_types(oids)
                for row in copy.rows():
                    decoded_rows.append(row)

        return _rows_to_table(
            columns, oids, decoded_rows, pa, numeric_as_float=numeric_as_float
        )


def _rows_to_table(
    columns: list[str],
    oids: list[int],
    rows: list[tuple[Any, ...]],
    pa: "Any",
    *,
    numeric_as_float: bool,
) -> "pa.Table":
    """Assemble decoded rows into a pyarrow.Table with an OID-derived schema."""
    # Transpose to columnar; empty result keeps one (empty) column per field.
    if rows:
        column_data = list(zip(*rows))
    else:
        column_data = [() for _ in columns]

    arrays = []
    fields = []
    for name, oid, data in zip(columns, oids, column_data):
        arrow_type = _oid_arrow_type(oid, pa)
        if (
            numeric_as_float
            and arrow_type is not None
            and pa.types.is_decimal(arrow_type)
        ):
            arrow_type = pa.float64()
            values: Any = [None if v is None else float(v) for v in data]
        else:
            values = list(data)

        try:
            array = pa.array(values, type=arrow_type)
            field_type = array.type
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
            # Unknown / unmappable OID, or a value that does not fit the inferred
            # type: fall back to a string column so the data round-trips and the
            # gap is visible, rather than silently dropping or mis-typing it.
            logger.warning(
                "Arrow: column {!r} (oid {}) did not fit type {}; "
                "falling back to string ({}).",
                name,
                oid,
                arrow_type,
                exc,
            )
            array = pa.array(
                [None if v is None else str(v) for v in data], type=pa.string()
            )
            field_type = pa.string()

        arrays.append(array)
        fields.append(pa.field(name, field_type))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def default_connection() -> Any:
    """Open a psycopg connection from ``DATABASE_URL`` or ``PG*`` env vars.

    Follows the project database hard-rule: never fall back to a guessed
    localhost. ``DATABASE_URL`` wins; otherwise an empty conninfo lets libpq read
    ``PG*``, but only when at least ``PGDATABASE`` or ``PGHOST`` is set.

    Raises:
        RuntimeError: If no database is configured in the environment.
    """
    import os

    psycopg = _require_psycopg()

    url = os.environ.get("DATABASE_URL")
    if url:
        conninfo = url
    elif os.environ.get("PGDATABASE") or os.environ.get("PGHOST"):
        conninfo = ""
    else:
        raise RuntimeError(
            "No PostgreSQL configured for Arrow/Parquet export: set DATABASE_URL "
            "or PG* env vars (e.g. via direnv). Refusing to guess a localhost "
            "database."
        )
    return psycopg.connect(conninfo, autocommit=False)
