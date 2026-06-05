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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

DEFAULT_COUNT_LIKE_PREFIXES = ("COUNT", "SUM", "NUNIQUE", "N_", "EVENT_RATE")


def _is_count_like(name: object, prefixes: Sequence[str]) -> bool:
    """True if a feature name's leading aggregation token is count-like.

    Feature names render as ``AGG(entity.col|interval=W)``; the token before the
    first ``(`` is the uppercased primitive name (e.g. ``COUNT``, ``SUM``).
    """
    token = str(name).split("(")[0].strip().upper()
    return any(token == p or token.startswith(p) for p in prefixes)


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
    if measure_strategy not in ("none", "median", "mean"):
        raise ValueError(
            f"measure_strategy must be 'none', 'median', or 'mean'; "
            f"got {measure_strategy!r}."
        )

    result = df.copy()
    target_cols = (
        list(result.columns)
        if columns is None
        else [c for c in columns if c in result.columns]
    )

    # 1. Missing indicators — recorded BEFORE any fill, only where NULLs exist.
    if add_missing_indicators:
        indicators = {
            f"{c}__missing": result[c].isnull().astype(int)
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
