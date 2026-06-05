from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Import name -> pip package name, where they differ.
_PIP_NAMES = {
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "cv2": "opencv-python",
}


def _require(module_name: str) -> None:
    """Raise ImportError with a helpful, install-ready message if missing."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        pip_name = _PIP_NAMES.get(module_name, module_name)
        raise ImportError(
            f"'{module_name}' is required for this visualization. "
            f"Install it with: pip install {pip_name} "
            f"(or: uv sync --extra viz)"
        ) from None


def _get_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Extract the numeric feature matrix from a DataFrame."""
    return df[feature_cols].select_dtypes(include="number")


def _latest_slice(df: pd.DataFrame, as_of_col: str, as_of_date: str | None):
    """Return the rows for a single as-of date (the latest one if None).

    Returns a tuple ``(sliced_df, resolved_as_of_date)``.
    """
    if as_of_col not in df.columns:
        # Nothing to slice on; treat the whole frame as a single snapshot.
        return df, None
    if as_of_date is None:
        as_of_date = df[as_of_col].max()
    return df[df[as_of_col] == as_of_date], as_of_date


def _impute_median(matrix: pd.DataFrame) -> pd.DataFrame:
    """Median-impute a numeric matrix on a *local copy*.

    sklearn / UMAP / scipy reject NaN, so methods that feed them must impute.
    The original matrix is never mutated. All-NaN columns fall back to 0.
    """
    filled = matrix.fillna(matrix.median(numeric_only=True))
    # Columns that were entirely NaN still hold NaN after a median fill.
    return filled.fillna(0.0)


def _zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score; zero-variance columns map to 0 (no divide-by-zero)."""
    std = matrix.std(ddof=0)
    std = std.where(std != 0, 1.0)
    return (matrix - matrix.mean()) / std


def _top_by_variance(matrix: pd.DataFrame, n: int) -> list[str]:
    """Return the names of the ``n`` highest-variance columns (descending)."""
    variances = matrix.var(numeric_only=True).sort_values(ascending=False)
    return list(variances.head(n).index)
