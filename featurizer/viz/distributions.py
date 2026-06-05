from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import _get_feature_matrix, _require, _top_by_variance

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


def plot_feature_distributions(
    self,
    features: list[str] | None = None,
    kind: str = "violin",
    figsize: tuple[int, int] = (14, 8),
) -> matplotlib.figure.Figure:
    """Plot feature distributions across entities.

    Args:
        features: Features to plot. If None, selects top 12 by variance.
        kind: Plot type ('violin', 'box', 'hist').
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    _require("seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = _get_feature_matrix(self.df, self.feature_cols)
    if features is None:
        features = _top_by_variance(matrix, 12)
    else:
        features = [f for f in features if f in matrix.columns]
    if not features:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No numeric features to plot", ha="center", va="center")
        return fig

    subset = matrix[features]
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "hist":
        subset.plot.hist(ax=ax, bins=30, alpha=0.5)
        ax.set_xlabel("Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    else:
        melted = subset.melt(var_name="feature", value_name="value").dropna()
        if kind == "box":
            sns.boxplot(data=melted, x="feature", y="value", ax=ax)
        else:  # default: violin
            sns.violinplot(data=melted, x="feature", y="value", ax=ax)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)

    ax.set_title(f"Feature Distributions ({kind})")
    plt.tight_layout()
    return fig


def feature_summary_table(self) -> pd.DataFrame:
    """Generate summary statistics table for all features.

    Returns:
        DataFrame (one row per numeric feature) with columns
        ``mean``, ``std``, ``skewness``, ``pct_missing``.
    """
    _require("pandas")
    import pandas as pd

    matrix = _get_feature_matrix(self.df, self.feature_cols)
    summary = pd.DataFrame(
        {
            "mean": matrix.mean(),
            "std": matrix.std(),
            "skewness": matrix.skew(),
            "pct_missing": matrix.isnull().mean() * 100.0,
        }
    )
    summary.index.name = "feature"
    return summary
