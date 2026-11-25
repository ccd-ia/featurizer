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
   PostgreSQL with binary ``COPY`` and decode it into Arrow in bounded batches.

Memory model (issue #95). The query runs once: its column names and type OIDs
come from a Parse/Describe of the statement, which executes nothing. ``COPY``
rows are decoded into Python objects (psycopg's per-OID loaders, about 30-40
bytes per value) at most ``batch_values`` values at a time, and each batch is
turned into Arrow arrays before the next is read. :meth:`ArrowExporter.to_arrow`
therefore peaks at the Arrow table plus one batch, and
:meth:`ArrowExporter.write_parquet` at one batch plus one buffered row group.
Two kinds of column are the exception and are held as Python objects until the
end, because their Arrow type depends on the values: ``numeric`` with
``numeric_as_float=False``, and any OID outside :func:`_oid_arrow_type`'s map.

This module is optional: it needs the ``[parquet]`` extra (``pyarrow``).
``psycopg`` is a core dependency, but the imports are kept lazy so importing
``featurizer`` never pulls Arrow in.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, TypeVar

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

_T = TypeVar("_T")

#: Values (rows x columns) decoded into Python objects before they are turned
#: into Arrow arrays. At ~40 bytes per value the default holds about 160 MB of
#: Python objects at once, whatever the size of the result.
DEFAULT_BATCH_VALUES = 4_000_000

#: Arrow bytes buffered before :meth:`ArrowExporter.write_parquet` writes a row
#: group. One decode batch is far below a useful Parquet row group, so batches
#: are grouped up to this size first.
_ROW_GROUP_BYTES = 128 * 2**20

_NUMERIC_OID = 1700
_UUID_OID = 2950

#: An Arrow type plus an optional per-value conversion, for a column whose Arrow
#: type is fixed by its OID alone.
_FixedType = Tuple["pa.DataType", Optional[Callable[[Any], Any]]]


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


def _fixed_type(oid: int, pa: "Any", numeric_as_float: bool) -> Optional[_FixedType]:
    """The column's Arrow type and value conversion, when the OID alone fixes them.

    ``None`` means the type depends on the values, so the column is built once
    every row is in (:func:`_build_column`): an OID outside the map is inferred
    by pyarrow, and an exact ``numeric`` falls back to string when a value does
    not fit ``decimal128(38, 9)``. Every other mapped type accepts every value
    psycopg's loader returns for it, so its batches can be built as they come.
    A ``uuid`` is mapped to string, and psycopg loads ``uuid.UUID`` objects, so
    they are converted with ``str`` (before #95 the same strings came out of the
    string fallback).
    """
    if oid == _NUMERIC_OID:
        return (pa.float64(), float) if numeric_as_float else None
    if oid == _UUID_OID:
        return pa.string(), str
    arrow_type = _oid_arrow_type(oid, pa)
    return None if arrow_type is None else (arrow_type, None)


class ArrowExporter:
    """Runs a rendered query and decodes the result into Arrow, no pandas hop.

    The result set is streamed out of PostgreSQL via binary ``COPY`` and decoded
    with psycopg's own per-OID loaders (so a SQL NULL becomes a Python ``None``,
    never ``NaN``), ``batch_values`` values at a time, into Arrow arrays with a
    schema derived from the statement's description. See the module docstring
    for the memory model.
    """

    def __init__(
        self,
        connection_factory: Optional[Callable[[], Any]] = None,
        *,
        batch_values: int = DEFAULT_BATCH_VALUES,
    ) -> None:
        """Initialize the exporter.

        Args:
            connection_factory: Callable returning an open psycopg connection.
                Defaults to a connection built from ``DATABASE_URL`` / ``PG*``
                via :func:`featurizer.arrow.default_connection`. The integration
                harness passes an existing connection (so session ``TEMP`` tables
                referenced by the query resolve) by supplying its own factory or
                using :meth:`to_arrow`'s ``connection`` argument.
            batch_values: How many values (rows x columns) are decoded into
                Python objects before they become Arrow arrays. Peak memory
                grows with this, not with the size of the result. A batch holds
                at least one row.

        Raises:
            ValueError: If ``batch_values`` is below 1.
        """
        if batch_values < 1:
            raise ValueError(f"batch_values must be at least 1, got {batch_values}.")
        self._connection_factory = connection_factory
        self.batch_values = batch_values

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
            leading columns (no index), and SQL NULLs are Arrow nulls. Each
            column is chunked by decode batch.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
            RuntimeError: If the database rejects the query. The full SQL is
                logged at error level so the failing CTE can be traced back to
                the planner builder that emitted it (same contract as
                :meth:`QueryExecutor.to_dataframe`).
        """
        pa = _require_pyarrow()
        return self._run(
            query,
            connection,
            lambda conn: self._decode_table(conn, query, pa, numeric_as_float),
        )

    def write_parquet(
        self,
        query: str,
        path: str,
        *,
        connection: Optional[Any] = None,
        numeric_as_float: bool = True,
    ) -> None:
        """Execute ``query`` and write the result to a Parquet file at ``path``.

        Batches are written as they are decoded, so the matrix is never held in
        memory, unless a column's type depends on its values (see the module
        docstring); then the table is built first and written whole. The file
        appears at ``path`` only once it is complete: it is written beside it
        under a temporary name and renamed, so a failed export leaves no
        partial file that would read as a valid, shorter matrix.

        Args:
            query: SQL query string to execute (the rendered featurizer query).
            path: Destination ``.parquet`` file. Its directory must exist.
            connection: See :meth:`to_arrow`.
            numeric_as_float: See :meth:`to_arrow`.

        Raises:
            ImportError: If pyarrow (the ``[parquet]`` extra) is not installed.
            RuntimeError: If the database rejects the query or the write fails.
        """
        pa = _require_pyarrow()
        return self._run(
            query,
            connection,
            lambda conn: self._write_parquet(conn, query, path, pa, numeric_as_float),
        )

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

    def _run(
        self, query: str, connection: Optional[Any], work: Callable[[Any], _T]
    ) -> _T:
        """Run ``work`` on the given or a fresh connection, logging the SQL on failure."""
        own_connection = connection is None
        conn = connection if connection is not None else self._open_connection()
        try:
            return work(conn)
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

    def _batch_rows(self, n_columns: int) -> int:
        return max(1, self.batch_values // max(1, n_columns))

    def _decode_table(
        self, conn: Any, query: str, pa: "Any", numeric_as_float: bool
    ) -> "pa.Table":
        """Describe the query, COPY it out in batches, and assemble the table."""
        columns, oids = _describe(conn, query)
        builder = _TableBuilder(columns, oids, pa, numeric_as_float)
        _copy_batches(conn, query, oids, self._batch_rows(len(columns)), builder.add)
        return builder.table()

    def _write_parquet(
        self, conn: Any, query: str, path: str, pa: "Any", numeric_as_float: bool
    ) -> None:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

        columns, oids = _describe(conn, query)
        builder = _TableBuilder(columns, oids, pa, numeric_as_float)
        target = os.fspath(path)
        fd, partial = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(target)),
            prefix=f".{os.path.basename(target)}.",
            suffix=".partial",
        )
        os.close(fd)
        batch_rows = self._batch_rows(len(columns))
        try:
            if builder.streamable:
                _stream_parquet(conn, query, oids, batch_rows, builder, partial, pa)
            else:
                _copy_batches(conn, query, oids, batch_rows, builder.add)
                pq.write_table(builder.table(), partial)
            os.replace(partial, target)
        except BaseException:
            os.unlink(partial)
            raise


def _stream_parquet(
    conn: Any,
    query: str,
    oids: list[int],
    batch_rows: int,
    builder: "_TableBuilder",
    path: str,
    pa: "Any",
) -> None:
    """COPY ``query`` into a Parquet file at ``path``, one row group per ~128 MB."""
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

    schema = builder.schema()
    buffered: list[Any] = []
    buffered_bytes = 0
    with pq.ParquetWriter(path, schema) as writer:

        def flush() -> None:
            nonlocal buffered_bytes
            if buffered:
                writer.write_table(pa.Table.from_batches(buffered, schema))
                buffered.clear()
                buffered_bytes = 0

        def add(rows: list[tuple[Any, ...]]) -> None:
            nonlocal buffered_bytes
            batch = pa.RecordBatch.from_arrays(builder.arrays(rows), schema=schema)
            buffered.append(batch)
            buffered_bytes += batch.nbytes
            if buffered_bytes >= _ROW_GROUP_BYTES:
                flush()

        _copy_batches(conn, query, oids, batch_rows, add)
        flush()


def _describe(conn: Any, query: str) -> tuple[list[str], list[int]]:
    """Column names and type OIDs of ``query``, without executing it.

    COPY's binary stream carries no column metadata, so the names and the OIDs
    the loaders need come from libpq's Parse + Describe of the unnamed
    statement: the server parses and analyzes the query but neither plans nor
    runs it. (Before #95 this was a full ``cursor.execute(query)``: the query
    ran twice, and libpq buffered the whole text-format result client-side.)
    """
    psycopg = _require_psycopg()
    encoding = conn.info.encoding
    ok = psycopg.pq.ExecStatus.COMMAND_OK
    with conn.lock:
        prepared = conn.pgconn.prepare(b"", query.encode(encoding))
        if prepared.status != ok:
            raise psycopg.errors.error_from_result(prepared, encoding=encoding)
        described = conn.pgconn.describe_prepared(b"")
        if described.status != ok:
            raise psycopg.errors.error_from_result(described, encoding=encoding)
    names = [
        (described.fname(i) or b"").decode(encoding) for i in range(described.nfields)
    ]
    oids = [described.ftype(i) for i in range(described.nfields)]
    return names, oids


def _copy_batches(
    conn: Any,
    query: str,
    oids: list[int],
    batch_rows: int,
    consume: Callable[[list[tuple[Any, ...]]], None],
) -> None:
    """Stream ``query`` out with binary COPY, handing ``batch_rows`` rows at a time to ``consume``."""
    copy_sql = f"copy ({query}) to stdout (format binary)"
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        copy.set_types(oids)
        batch: list[tuple[Any, ...]] = []
        for row in copy.rows():
            batch.append(row)
            if len(batch) == batch_rows:
                consume(batch)
                batch = []
        if batch:
            consume(batch)


class _TableBuilder:
    """Turns decoded row batches into Arrow, one column at a time.

    A column with a fixed type (:func:`_fixed_type`) gets one Arrow chunk per
    batch, and the batch's Python objects are dropped. A column whose type
    depends on its values keeps them until :meth:`table`, which builds it
    exactly as the pre-#95 whole-result path did.
    """

    def __init__(
        self, columns: list[str], oids: list[int], pa: "Any", numeric_as_float: bool
    ) -> None:
        self._columns = columns
        self._oids = oids
        self._pa = pa
        self._fixed = [_fixed_type(oid, pa, numeric_as_float) for oid in oids]
        self._chunks: list[list[Any]] = [[] for _ in columns]
        self._deferred: dict[int, list[Any]] = {
            i: [] for i, fixed in enumerate(self._fixed) if fixed is None
        }

    @property
    def streamable(self) -> bool:
        """Every column's type is known before the first row: batches can be written."""
        return not self._deferred

    def schema(self) -> "pa.Schema":
        """The table's schema; only defined when :attr:`streamable`."""
        return self._pa.schema(
            [
                self._pa.field(name, fixed[0])
                for name, fixed in zip(self._columns, self._fixed)
                if fixed is not None
            ]
        )

    def arrays(self, rows: list[tuple[Any, ...]]) -> list[Any]:
        """One Arrow array per column for a batch; only when :attr:`streamable`."""
        return [
            _fixed_array(data, fixed, self._pa)
            for data, fixed in zip(zip(*rows), self._fixed)
            if fixed is not None
        ]

    def add(self, rows: list[tuple[Any, ...]]) -> None:
        """Convert one batch: fixed columns to an Arrow chunk, the rest kept aside."""
        for i, data in enumerate(zip(*rows)):
            fixed = self._fixed[i]
            if fixed is None:
                self._deferred[i].extend(data)
            else:
                self._chunks[i].append(_fixed_array(data, fixed, self._pa))

    def table(self) -> "pa.Table":
        """Assemble every column; an empty result keeps one empty column per field."""
        pa = self._pa
        arrays: list[Any] = []
        fields: list[Any] = []
        for i, (name, oid) in enumerate(zip(self._columns, self._oids)):
            fixed = self._fixed[i]
            if fixed is None:
                array = _build_column(name, oid, self._deferred.pop(i), pa)
            else:
                array = pa.chunked_array(self._chunks[i], type=fixed[0])
                self._chunks[i] = []
            arrays.append(array)
            fields.append(pa.field(name, array.type))
        return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _fixed_array(data: tuple[Any, ...], fixed: _FixedType, pa: "Any") -> Any:
    """One batch of a fixed-type column as an Arrow array (NULL -> Arrow null)."""
    arrow_type, convert = fixed
    if convert is None:
        return pa.array(data, type=arrow_type)
    return pa.array([None if v is None else convert(v) for v in data], type=arrow_type)


def _build_column(name: str, oid: int, values: list[Any], pa: "Any") -> Any:
    """Build a value-typed column whole: its mapped type, else pyarrow's inference.

    A value that does not fit falls back to a string column, so the data
    round-trips and the gap is visible, rather than silently dropping or
    mis-typing it.
    """
    arrow_type = _oid_arrow_type(oid, pa)
    try:
        return pa.array(values, type=arrow_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
        logger.warning(
            "Arrow: column {!r} (oid {}) did not fit type {}; "
            "falling back to string ({}).",
            name,
            oid,
            arrow_type,
            exc,
        )
        return pa.array(
            [None if v is None else str(v) for v in values], type=pa.string()
        )


def default_connection() -> Any:
    """Open a psycopg connection from ``DATABASE_URL`` or ``PG*`` env vars.

    Follows the project database hard-rule: never fall back to a guessed
    localhost. ``DATABASE_URL`` wins; otherwise an empty conninfo lets libpq read
    ``PG*``, but only when at least ``PGDATABASE`` or ``PGHOST`` is set.

    Raises:
        RuntimeError: If no database is configured in the environment.
    """
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
