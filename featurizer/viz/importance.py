from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import _get_feature_matrix, _impute_median, _require

if TYPE_CHECKING:
    import matplotlib.figure


def plot_feature_importance(
    self,
    target_col: str,
    kind: str = "mutual_info",
    top_n: int = 30,
    figsize: tuple[int, int] = (12, 10),
) -> matplotlib.figure.Figure:
    """Plot ranked feature importance.

    Args:
        target_col: Target variable column name.
        kind: Method ('mutual_info', 'f_classif', 'f_regression').
        top_n: Number of top features to show.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    _require("sklearn")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.feature_selection import (
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
    )

    if target_col not in self.df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame columns.")

    matrix = _get_feature_matrix(self.df, self.feature_cols)
    X = matrix.drop(columns=[target_col], errors="ignore")
    y_raw = self.df[target_col]

    # Drop rows with a missing target; median-impute the features locally.
    mask = y_raw.notna()
    X = _impute_median(X[mask.to_numpy()])
    y_raw = y_raw[mask]

    # Decide regression vs classification.
    y_numeric = pd.api.types.is_numeric_dtype(y_raw)
    if kind == "f_regression":
        regression = True
    elif kind == "f_classif":
        regression = False
    else:  # mutual_info: infer from the target
        regression = y_numeric and y_raw.nunique() > 20

    y = y_raw if regression else pd.factorize(y_raw)[0]

    scorers = {
        ("mutual_info", True): mutual_info_regression,
        ("mutual_info", False): mutual_info_classif,
        ("f_regression", True): f_regression,
        ("f_classif", False): f_classif,
    }
    key = (kind if kind in ("f_regression", "f_classif") else "mutual_info", regression)
    scorer = scorers[key]

    raw = scorer(X, y)
    scores = raw[0] if kind.startswith("f_") else raw  # f_* returns (F, pvalue)
    ranked = pd.Series(scores, index=X.columns).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1], color="steelblue")
    ax.set_title(f"Feature Importance ({kind}, target={target_col})")
    ax.set_xlabel("Score")
    plt.tight_layout()
    return fig


def plot_feature_variance(
    self,
    top_n: int = 50,
    figsize: tuple[int, int] = (12, 10),
) -> matplotlib.figure.Figure:
    """Plot feature variance ranking (no target needed).

    Args:
        top_n: Number of top features to show.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    import matplotlib.pyplot as plt

    matrix = _get_feature_matrix(self.df, self.feature_cols)
    ranked = matrix.var().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1], color="darkorange")
    ax.set_title(f"Feature Variance (top {top_n})")
    ax.set_xlabel("Variance")
    plt.tight_layout()
    return fig
