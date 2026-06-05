from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import _get_feature_matrix, _require, _top_by_variance, _zscore

if TYPE_CHECKING:
    import matplotlib.figure


def plot_feature_timeseries(
    self,
    entity_id: str | int,
    features: list[str] | None = None,
    normalize: bool = False,
    figsize: tuple[int, int] = (14, 8),
) -> matplotlib.figure.Figure:
    """Plot feature values over time for a single entity.

    Args:
        entity_id: Entity to plot.
        features: Features to plot. If None, selects top 8 by variance.
        normalize: Z-score normalize features for comparison.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    import matplotlib.pyplot as plt

    sub = self.df[self.df[self.entity_col] == entity_id].sort_values(self.as_of_col)
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, f"No rows for entity {entity_id}", ha="center", va="center")
        return fig

    matrix = _get_feature_matrix(sub, self.feature_cols)
    if features is None:
        features = _top_by_variance(matrix, 8)
    else:
        features = [f for f in features if f in matrix.columns]
    if not features:
        ax.text(0.5, 0.5, "No numeric features to plot", ha="center", va="center")
        return fig

    data = matrix[features]
    if normalize:
        data = _zscore(data)

    x = sub[self.as_of_col].to_numpy()
    for col in features:
        ax.plot(
            x, data[col].to_numpy(), marker="o", markersize=3, linewidth=1, label=col
        )

    ax.set_title(f"Feature Time Series (entity={entity_id})")
    ax.set_xlabel(self.as_of_col)
    ax.set_ylabel("z-score" if normalize else "value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_entity_feature_heatmap(
    self,
    entity_id: str | int,
    figsize: tuple[int, int] = (16, 10),
) -> matplotlib.figure.Figure:
    """Plot features x time z-scored heatmap for a single entity.

    Args:
        entity_id: Entity to plot.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    _require("seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = self.df[self.df[self.entity_col] == entity_id].sort_values(self.as_of_col)
    fig, ax = plt.subplots(figsize=figsize)
    if sub.empty:
        ax.text(0.5, 0.5, f"No rows for entity {entity_id}", ha="center", va="center")
        return fig

    matrix = _get_feature_matrix(sub, self.feature_cols)
    # z-score each feature across this entity's timepoints, then features-as-rows.
    data = _zscore(matrix).T
    data.columns = sub[self.as_of_col].astype(str).to_numpy()

    sns.heatmap(data, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title(f"Feature x Time (entity={entity_id})")
    ax.set_xlabel(self.as_of_col)
    ax.set_ylabel("Feature")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout()
    return fig
