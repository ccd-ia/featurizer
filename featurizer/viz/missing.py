from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import _require

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


def plot_missing_heatmap(
    self,
    as_of_date: str | None = None,
    figsize: tuple[int, int] = (16, 10),
    cmap: str = "YlOrRd",
    max_entities: int = 50,
) -> matplotlib.figure.Figure:
    """Plot binary heatmap of missing values (entities x features).

    Args:
        as_of_date: Filter to a specific as_of_date. If None, uses latest.
        figsize: Figure size.
        cmap: Colormap for the heatmap.
        max_entities: Maximum entities to show (samples if exceeded).

    Returns:
        matplotlib Figure.
    """
    _require("seaborn")
    _require("matplotlib")
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = self.df.copy()
    if as_of_date is not None:
        df = df[df[self.as_of_col] == as_of_date]
    else:
        latest = df[self.as_of_col].max()
        df = df[df[self.as_of_col] == latest]
        as_of_date = str(latest)

    if self.entity_col in df.columns:
        df = df.set_index(self.entity_col)

    feature_df = df[self.feature_cols]
    if len(feature_df) > max_entities:
        feature_df = feature_df.sample(max_entities, random_state=42)

    missing = feature_df.isnull().astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        missing,
        cmap=cmap,
        cbar_kws={"label": "Missing (1=yes)"},
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )
    ax.set_title(f"Missing Data Pattern (as_of_date={as_of_date})")
    ax.set_xlabel("Features")
    ax.set_ylabel("Entities")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    return fig


def plot_missing_over_time(
    self,
    figsize: tuple[int, int] = (14, 8),
    top_n: int = 20,
) -> matplotlib.figure.Figure:
    """Plot percentage of missing values per feature across as_of_dates.

    Args:
        figsize: Figure size.
        top_n: Show only top N features with most missingness.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    import matplotlib.pyplot as plt

    grouped = self.df.groupby(self.as_of_col)[self.feature_cols].apply(
        lambda x: x.isnull().mean()
    )

    avg_missing = grouped.mean().sort_values(ascending=False)
    top_features = avg_missing.head(top_n).index.tolist()

    if not top_features:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No missing data found", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    subset = grouped[top_features]
    subset.plot(ax=ax, linewidth=1.5, alpha=0.8)
    ax.set_title(f"Missing Data Over Time (top {top_n} features)")
    ax.set_xlabel("As-of Date")
    ax.set_ylabel("Fraction Missing")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig
