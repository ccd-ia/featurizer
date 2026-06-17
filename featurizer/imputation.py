"""Opt-in imputation for featurizer output matrices.

Featurizer never imputes inside the generated SQL: a NULL in the
``(as_of_date × entity)`` matrix means "no qualifying events in the window",
and that missingness pattern is itself predictive signal. This module provides
an *explicit*, opt-in fill that respects two rules:

1. **count-vs-measure.** Count-like features (``COUNT``/``SUM``/``NUNIQUE``/…)
   have a meaningful structural zero — no events genuinely means 0 — so they are
   filled with 0. Measures (``AVG``/``MEDIAN``/``STDDEV``/percentiles/recency/…)
   have no basis for a fill and stay NULL unless an explicit ``measure_strategy``
   is requested.
2. **never lose the signal.** When ``add_missing_indicators`` is set, a
   ``<feature>__missing`` 0/1 column is emitted for every column that had NULLs
   *before* any value is filled.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa

DEFAULT_COUNT_LIKE_PREFIXES = ("COUNT", "SUM", "NUNIQUE", "N_", "EVENT_RATE")

#: Suffix of the 0/1 indicator columns emitted for features that had NULLs.
#: This is a **stable contract**: downstream adapters may rely on the name
#: ``f"{feature}{MISSING_INDICATOR_SUFFIX}"`` to locate the missingness flags on
#: both the pandas (:func:`impute_features`) and Arrow (:func:`impute_arrow`)
#: output paths. Do not change it without bumping the public contract.
MISSING_INDICATOR_SUFFIX = "__missing"


def _is_count_like(name: object, prefixes: Sequence[str]) -> bool:
    """True if a feature name's leading aggregation token is count-like.

    Feature names render as ``AGG(entity.col|interval=W)``; the token before the
    first ``(`` is the uppercased primitive name (e.g. ``COUNT``, ``SUM``).
    """
    token = str(name).split("(")[0].strip().upper()
    return any(token == p or token.startswith(p) for p in prefixes)


def _validate_measure_strategy(measure_strategy: str) -> None:
    if measure_strategy not in ("none", "median", "mean"):
        raise ValueError(
            f"measure_strategy must be 'none', 'median', or 'mean'; "
            f"got {measure_strategy!r}."
        )


def guard_full_matrix_fit(
    measure_strategy: str, *, allow_full_matrix_fit: bool, caller: str
) -> None:
    """Refuse a whole-matrix measure fit unless explicitly authorized.

    ``measure_strategy in {"mean", "median"}`` computes the fill value over the
    *entire* returned matrix — every ``as_of_date`` and the whole cohort,
    including validation/test rows. Fitting an imputation statistic over data
    that includes the evaluation rows is temporal leakage (see ADR-0001: any
    model an estimator needs is fit only on rows knowable as-of the cutoff).

    The engine-driven paths (:meth:`Featurizer.to_dataframe`,
    :meth:`Featurizer.to_parquet`, :meth:`Featurizer.to_arrow`) call this so the
    leaky branch is unreachable by accident: it requires
    ``allow_full_matrix_fit=True`` AND always emits a runtime warning when used.

    Args:
        measure_strategy: The requested measure fill (``none``/``mean``/``median``).
        allow_full_matrix_fit: Caller's explicit opt-in.
        caller: Name of the calling method, for the message.

    Raises:
        ValueError: If a fitting strategy is requested without the opt-in.
    """
    _validate_measure_strategy(measure_strategy)
    if measure_strategy == "none":
        return
    if not allow_full_matrix_fit:
        raise ValueError(
            f"{caller}(measure_strategy={measure_strategy!r}) fits the fill value "
            "over the WHOLE returned matrix (all as_of_dates and the entire "
            "cohort, including validation/test rows). That is temporal leakage "
            "(ADR-0001). If you really want a quick, leaky baseline, pass "
            "allow_full_matrix_fit=True; otherwise fit the imputer on your "
            "training split only and apply it downstream."
        )
    warnings.warn(
        f"{caller}: measure_strategy={measure_strategy!r} with "
        "allow_full_matrix_fit=True fits over the entire returned matrix "
        "(including validation/test rows) — temporal leakage (ADR-0001). Use "
        "only for a throwaway baseline; never for a model you will evaluate.",
        stacklevel=3,
    )


def impute_features(
    df: pd.DataFrame,
    *,
    count_fill_zero: bool = True,
    measure_strategy: str = "none",
    add_missing_indicators: bool = True,
    count_like_prefixes: Sequence[str] = DEFAULT_COUNT_LIKE_PREFIXES,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Impute a feature matrix, preserving the missingness signal.

    Args:
        df: Feature matrix. The entity/as-of keys are expected to live in the
            index (the ``Featurizer.to_dataframe`` shape); pass ``columns`` to
            restrict which columns are treated as features otherwise.
        count_fill_zero: Fill count-like features with 0 (structural zero).
        measure_strategy: Fill for non-count numeric features — one of
            ``"none"`` (leave NULL), ``"median"``, ``"mean"``.
        add_missing_indicators: Emit a ``<feature>__missing`` 0/1 column for
            each column that had NULLs, recorded before filling.
        count_like_prefixes: Aggregation-name prefixes treated as count-like.
        columns: Explicit feature columns to operate on (default: all columns).

    Returns:
        A new DataFrame (the input is never mutated).
    """
    _validate_measure_strategy(measure_strategy)

    result = df.copy()
    target_cols = (
        list(result.columns)
        if columns is None
        else [c for c in columns if c in result.columns]
    )

    # 1. Missing indicators — recorded BEFORE any fill, only where NULLs exist.
    if add_missing_indicators:
        indicators = {
            f"{c}{MISSING_INDICATOR_SUFFIX}": result[c].isnull().astype(int)
            for c in target_cols
            if bool(result[c].isnull().to_numpy().any())
        }
        for name, series in indicators.items():
            result[name] = series

    # 2. Fill — count-like to 0; measures left NULL unless a strategy is given.
    count_cols = [c for c in target_cols if _is_count_like(c, count_like_prefixes)]
    measure_cols = [c for c in target_cols if c not in count_cols]

    if count_fill_zero and count_cols:
        result[count_cols] = result[count_cols].fillna(0)

    if measure_strategy != "none" and measure_cols:
        numeric_measures = result[measure_cols].select_dtypes(include="number").columns
        if len(numeric_measures):
            fill = (
                result[numeric_measures].median()
                if measure_strategy == "median"
                else result[numeric_measures].mean()
            )
            result[numeric_measures] = result[numeric_measures].fillna(fill)

    return result


def impute_arrow(
    table: "pa.Table",
    *,
    key_columns: Sequence[str] = (),
    count_fill_zero: bool = True,
    measure_strategy: str = "none",
    add_missing_indicators: bool = True,
    count_like_prefixes: Sequence[str] = DEFAULT_COUNT_LIKE_PREFIXES,
) -> "pa.Table":
    """Arrow-native mirror of :func:`impute_features`.

    Applies the same NULL-preserving contract to a :class:`pyarrow.Table` so the
    Parquet/Arrow output path behaves identically to ``to_dataframe(impute=...)``:
    count-like features are filled with the structural zero, measures stay null
    unless an explicit ``measure_strategy`` is requested, and a stable
    ``<feature>__missing`` 0/1 column is emitted (recorded *before* any fill) for
    every feature column that had nulls.

    Unlike the pandas path (where ``as_of_date`` and the target id live in the
    index), an Arrow table carries the keys as ordinary columns; pass them in
    ``key_columns`` so they are never treated as features.

    The prefix logic and naming are shared with :func:`impute_features` — the
    count-like classification (:func:`_is_count_like`) and the
    :data:`MISSING_INDICATOR_SUFFIX` are not duplicated.

    Args:
        table: The feature matrix as a pyarrow.Table (e.g. from
            :class:`featurizer.arrow.ArrowExporter`).
        key_columns: Columns to leave untouched (``as_of_date``, target id).
        count_fill_zero: Fill count-like features with 0 (structural zero).
        measure_strategy: Fill for non-count numeric features — one of
            ``"none"`` (leave null), ``"median"``, ``"mean"``. NOTE: ``mean`` /
            ``median`` fit over the whole table; the engine path gates this
            behind :func:`guard_full_matrix_fit` (ADR-0001 leakage rule).
        add_missing_indicators: Emit a ``<feature>__missing`` 0/1 column for each
            feature column that had nulls, recorded before filling.
        count_like_prefixes: Aggregation-name prefixes treated as count-like.

    Returns:
        A new pyarrow.Table (the input is never mutated).
    """
    import pyarrow as pa  # local import: only needed on the Arrow path
    import pyarrow.compute as pc

    _validate_measure_strategy(measure_strategy)

    keys = set(key_columns)
    feature_cols = [n for n in table.column_names if n not in keys]

    columns: dict[str, Any] = {n: table.column(n) for n in table.column_names}
    appended: list[tuple[str, Any]] = []

    for name in feature_cols:
        col = table.column(name)
        had_nulls = col.null_count > 0

        # 1. Missing indicator — recorded BEFORE any fill, only where nulls exist.
        if add_missing_indicators and had_nulls:
            # pyarrow.compute functions are generated at import time, so the type
            # checker cannot see them statically (matches the [bridge] idiom).
            is_null = pc.is_null(col)  # pyright: ignore[reportAttributeAccessIssue]
            indicator = pc.cast(is_null, pa.int64())
            appended.append((f"{name}{MISSING_INDICATOR_SUFFIX}", indicator))

        if not had_nulls:
            continue

        # 2. Fill — count-like to 0; measures left null unless a strategy is set.
        if count_fill_zero and _is_count_like(name, count_like_prefixes):
            columns[name] = _fill_arrow(col, pa.scalar(0), pa)
        elif measure_strategy != "none" and pa.types.is_floating(col.type):
            stat = (
                pc.mean(col)  # pyright: ignore[reportAttributeAccessIssue]
                if measure_strategy == "mean"
                else _arrow_median(col, pc)
            )
            if stat.is_valid:
                columns[name] = _fill_arrow(col, stat, pa)

    arrays = [columns[n] for n in table.column_names] + [a for _, a in appended]
    names = list(table.column_names) + [n for n, _ in appended]
    return pa.table(dict(zip(names, arrays)))


def _fill_arrow(column: "Any", value: "Any", pa: "Any") -> "Any":
    """Fill nulls in a (possibly chunked) Arrow column, casting the fill to its type."""
    import pyarrow.compute as pc

    filled = pc.fill_null(column, pc.cast(value, column.type))
    # Keep the column shape (ChunkedArray) consistent for pa.table assembly.
    return filled


def _arrow_median(column: "Any", pc: "Any") -> "Any":
    """Median of an Arrow numeric column as a scalar (nulls skipped)."""
    approx = pc.approximate_median(column)
    return approx
